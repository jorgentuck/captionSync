// Package subtitle reads SRT, WebVTT and ASS/SSA files and rewrites ONLY their timestamps.
//
// Files are handled line by line as raw bytes: the text, styling, encoding, BOM and line
// endings are written back exactly as they were - the only bytes that change are the times.
package subtitle

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

type Cue struct {
	Start, End float64 // seconds
	Text       string  // UTF-8
}

type entry struct {
	line       int
	start, end float64
	cue        bool // false for ASS "Comment:" lines (shifted, but not used for matching)
	text       string
	// ASS only
	prefix       string
	fields       []string
	startF, endF int
	srtRest      string
	bom          bool // the timing line began with a UTF-8 BOM (first line of the file)
}

type File struct {
	Path    string
	Format  string // srt | vtt | ass
	lines   []string
	entries []entry
}

var timingRe = regexp.MustCompile(`^\s*((?:\d+:)?\d{1,2}:\d{1,2}[,.]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{1,2}[,.]\d{1,3})(.*)$`)

func Load(path string) (*File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	f := &File{Path: path}
	// Split on \n only: a line keeps its own trailing \r, so mixed line endings survive untouched.
	f.lines = strings.Split(string(raw), "\n")

	switch strings.ToLower(filepath.Ext(path)) {
	case ".ass", ".ssa":
		f.Format = "ass"
		err = f.parseASS()
	case ".vtt":
		f.Format = "vtt"
		f.parseTimed()
	default:
		f.Format = "srt"
		f.parseTimed()
	}
	if err != nil {
		return nil, err
	}
	if len(f.entries) == 0 {
		return nil, fmt.Errorf("no subtitle cues found")
	}
	return f, nil
}

// --- SRT / VTT -------------------------------------------------------------

// parseClock reads H:MM:SS.frac or MM:SS.frac. With intMillis (SRT/VTT) the fraction is an integer
// count of milliseconds, as ffmpeg/VLC/libass read it (",5" = 5 ms); otherwise (ASS) it is decimal.
func parseClock(s string, intMillis bool) (float64, bool) {
	s = strings.ReplaceAll(s, ",", ".")
	parts := strings.Split(s, ":")
	var h, m float64
	var sec string
	switch len(parts) {
	case 3:
		hh, err1 := strconv.Atoi(parts[0])
		mm, err2 := strconv.Atoi(parts[1])
		if err1 != nil || err2 != nil {
			return 0, false
		}
		h, m, sec = float64(hh), float64(mm), parts[2]
	case 2:
		mm, err := strconv.Atoi(parts[0])
		if err != nil {
			return 0, false
		}
		m, sec = float64(mm), parts[1]
	default:
		return 0, false
	}
	if intMillis {
		whole, frac, _ := strings.Cut(sec, ".")
		si, err1 := strconv.Atoi(whole)
		fi, err2 := strconv.Atoi(frac)
		if err1 != nil || err2 != nil {
			return 0, false
		}
		return h*3600 + m*60 + float64(si) + float64(fi)/1000, true
	}
	sv, err := strconv.ParseFloat(sec, 64)
	if err != nil {
		return 0, false
	}
	return h*3600 + m*60 + sv, true
}

func (f *File) parseTimed() {
	for i := 0; i < len(f.lines); i++ {
		line, bom := strings.CutPrefix(f.lines[i], "\uFEFF")
		m := timingRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		st, ok1 := parseClock(m[1], true)
		en, ok2 := parseClock(m[2], true)
		if !ok1 || !ok2 {
			continue
		}
		var text []string
		j := i + 1
		for ; j < len(f.lines) && strings.TrimSpace(f.lines[j]) != ""; j++ {
			if timingRe.MatchString(f.lines[j]) {
				break
			}
			text = append(text, strings.TrimSuffix(f.lines[j], "\r"))
		}
		f.entries = append(f.entries, entry{line: i, start: st, end: en, cue: true,
			text: toUTF8(strings.Join(text, "\n")), srtRest: m[3], bom: bom})
		i = j - 1
	}
}

func fmtClock(t float64, sep string) string {
	ms := int64(math.Round(math.Max(0, t) * 1000))
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", ms/3600000, ms/60000%60, ms/1000%60, sep, ms%1000)
}

// --- ASS / SSA -------------------------------------------------------------

// parseASSClock reads H:MM:SS.cc; the fraction is an integer count of centiseconds, as libass and
// ffmpeg read it ("0:00:01.5" = 1.05 s).
func parseASSClock(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	whole, frac, found := strings.Cut(s, ".")
	if !found {
		return parseClock(s, false)
	}
	t, ok := parseClock(whole+".0", false)
	cs, err := strconv.Atoi(frac)
	if !ok || err != nil {
		return 0, false
	}
	return t + float64(cs)/100, true
}

func fmtASSClock(t float64) string {
	cs := int64(math.Round(math.Max(0, t) * 100))
	return fmt.Sprintf("%d:%02d:%02d.%02d", cs/360000, cs/6000%60, cs/100%60, cs%100)
}

// ASS event kinds: Dialogue lines are matched against speech; the others are only shifted along.
var assShiftOnly = map[string]bool{"comment": true, "picture": true, "sound": true, "movie": true, "command": true}

func (f *File) parseASS() error {
	inEvents := false
	var format []string
	for i, raw := range f.lines {
		line, eol := strings.CutSuffix(raw, "\r") // keep a CRLF line's \r out of the fields
		t := strings.TrimSpace(strings.TrimPrefix(line, "\uFEFF"))
		if strings.HasPrefix(t, "[") {
			inEvents = strings.EqualFold(t, "[Events]")
			continue
		}
		if !inEvents {
			continue
		}
		colon := strings.Index(line, ":")
		if colon < 0 {
			continue
		}
		kind := strings.ToLower(strings.TrimSpace(line[:colon]))
		body := line[colon+1:]
		if kind == "format" {
			format = nil
			for _, fld := range strings.Split(body, ",") {
				format = append(format, strings.ToLower(strings.TrimSpace(fld)))
			}
			for len(format) > 0 && format[len(format)-1] == "" { // "Format: ..., Text," trailing comma
				format = format[:len(format)-1]
			}
			continue
		}
		isDialogue := kind == "dialogue"
		if !isDialogue && !assShiftOnly[kind] {
			continue
		}
		if format == nil {
			format = []string{"layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"}
		}
		si, ei, ti := indexOf(format, "start"), indexOf(format, "end"), indexOf(format, "text")
		if si < 0 || ei < 0 {
			return fmt.Errorf("ASS Format line has no Start/End")
		}
		// Text may contain commas, so only a Text column that comes last absorbs the rest of the line.
		n := len(format)
		if ti >= 0 && ti != n-1 {
			n = -1
		}
		fields := strings.SplitN(body, ",", n)
		if si >= len(fields) || ei >= len(fields) {
			continue // not even the times are present: leave this line alone
		}
		text := ""
		if ti >= 0 && ti < len(fields) && len(fields) >= len(format) {
			text = fields[ti] // short (malformed) lines are still retimed, just not matched
		}
		st, ok1 := parseASSClock(fields[si])
		en, ok2 := parseASSClock(fields[ei])
		if !ok1 || !ok2 {
			continue
		}
		e := entry{line: i, start: st, end: en, cue: isDialogue,
			text: toUTF8(text), prefix: line[:colon+1], fields: fields, startF: si, endF: ei}
		if eol {
			e.srtRest = "\r" // ASS lines don't use srtRest otherwise: it carries the line ending back
		}
		f.entries = append(f.entries, e)
	}
	return nil
}

func indexOf(xs []string, s string) int {
	for i, x := range xs {
		if x == s {
			return i
		}
	}
	return -1
}

// --- public API -------------------------------------------------------------

// Cues returns the dialogue cues (used for matching against speech).
func (f *File) Cues() []Cue {
	out := make([]Cue, 0, len(f.entries))
	for _, e := range f.entries {
		if e.cue && strings.TrimSpace(e.text) != "" {
			out = append(out, Cue{Start: e.start, End: e.end, Text: e.text})
		}
	}
	return out
}

// Retime applies fn to every timed line and returns how many changed.
func (f *File) Retime(fn func(start, end float64) (float64, float64)) int {
	changed := 0
	for k := range f.entries {
		e := &f.entries[k]
		ns, ne := fn(e.start, e.end)
		ns = math.Max(0, ns)
		if ne <= ns {
			ne = ns + 0.01
		}
		var line string
		switch f.Format {
		case "ass":
			fields := append([]string(nil), e.fields...)
			fields[e.startF] = fmtASSClock(ns)
			fields[e.endF] = fmtASSClock(ne)
			line = e.prefix + strings.Join(fields, ",") + e.srtRest
		case "vtt":
			line = fmtClock(ns, ".") + " --> " + fmtClock(ne, ".") + e.srtRest
		default:
			line = fmtClock(ns, ",") + " --> " + fmtClock(ne, ",") + e.srtRest
		}
		if e.bom {
			line = "\uFEFF" + line
		}
		if line != f.lines[e.line] {
			changed++
			f.lines[e.line] = line
		}
	}
	return changed
}

func (f *File) Bytes() []byte { return []byte(strings.Join(f.lines, "\n")) }

// --- encoding ---------------------------------------------------------------

var cp1252 = [32]rune{
	'€', '\u0081', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u008d', 'Ž', '\u008f',
	'\u0090', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u009d', 'ž', 'Ÿ',
}

// toUTF8 converts legacy Windows-1252/Latin-1 text for matching (the file itself is never re-encoded).
func toUTF8(s string) string {
	s = strings.TrimPrefix(s, "\uFEFF")
	if utf8.ValidString(s) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c < 0x80:
			b.WriteByte(c)
		case c < 0xA0:
			b.WriteRune(cp1252[c-0x80])
		default:
			b.WriteRune(rune(c))
		}
	}
	return b.String()
}
