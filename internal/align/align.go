// Package align works out how far each subtitle cue is from the speech it belongs to.
//
//  1. Text anchors: each cue's words are fuzzy-matched against words transcribed from the
//     audio near the cue. A good, unambiguous match gives an anchor: subtitle time -> speech time.
//  2. Anchors are cleaned of outliers and split into segments that share a consistent offset.
//     This covers a single global offset, framerate drift, and jumps where an edit/ad break differs.
//  3. If the text does not match (e.g. subtitles in another language than the audio) we fall back
//     to correlating "someone is speaking" with "a subtitle is on screen" to find a global offset.
package align

import (
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// ---------------------------------------------------------------------------
// Text normalisation
// ---------------------------------------------------------------------------

var (
	tagsRe    = regexp.MustCompile(`\{[^}]*\}|<[^>]*>`)
	sdhRe     = regexp.MustCompile(`\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪?|#[^#\n]*#?`)
	speakerRe = regexp.MustCompile(`^\s*-?\s*[A-Z][A-Z0-9 .'\-]{1,30}:\s*`)
	wordRe    = regexp.MustCompile(`[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)?`)
	apos      = strings.NewReplacer("’", "'", "‘", "'", "`", "'")
)

// Tokenize lower-cases, strips tags/accents and (optionally) SDH like [door slams] or "JOHN:".
func Tokenize(text string, stripSDH bool) []string {
	text = strings.NewReplacer(`\N`, "\n", `\n`, "\n", `\h`, " ").Replace(text)
	text = tagsRe.ReplaceAllString(text, " ")
	if stripSDH {
		lines := strings.Split(text, "\n")
		for i, l := range lines {
			lines[i] = speakerRe.ReplaceAllString(l, "")
		}
		text = sdhRe.ReplaceAllString(strings.Join(lines, "\n"), " ")
	}
	text = apos.Replace(text)
	var b strings.Builder
	for _, r := range norm.NFKD.String(text) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return wordRe.FindAllString(b.String(), -1)
}

// Ratio is a 0-100 similarity score (normalised Indel distance, like rapidfuzz's fuzz.ratio).
func Ratio(a, b string) float64 {
	ra, rb := []rune(a), []rune(b)
	if len(ra)+len(rb) == 0 {
		return 100
	}
	if len(ra) == 0 || len(rb) == 0 {
		return 0
	}
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for i := 1; i <= len(ra); i++ {
		for j := 1; j <= len(rb); j++ {
			switch {
			case ra[i-1] == rb[j-1]:
				cur[j] = prev[j-1] + 1
			case prev[j] >= cur[j-1]:
				cur[j] = prev[j]
			default:
				cur[j] = cur[j-1]
			}
		}
		prev, cur = cur, prev
	}
	lcs := prev[len(rb)]
	return 100 * float64(2*lcs) / float64(len(ra)+len(rb))
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Cue struct {
	Start, End float64
	Tokens     []string
}

type Word struct {
	Text       string
	Start, End float64
	Reliable   bool // timestamp is trustworthy (e.g. start of a whisper segment)
}

type Anchor struct {
	SubTime, AudioTime, Score float64
}

func (a Anchor) Delta() float64 { return a.AudioTime - a.SubTime }

// Piece shifts cues whose start falls in [Start, End).
type Piece struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Offset  float64 `json:"offset"`
	Slope   float64 `json:"slope"`
	Ref     float64 `json:"ref"`
	Anchors int     `json:"anchors"`
}

func (p Piece) Shift(t float64) float64 { return p.Offset + p.Slope*(t-p.Ref) }

type Mapping struct {
	Scale       float64 `json:"scale"` // framerate correction, applied before the pieces (1 = none)
	Pieces      []Piece `json:"pieces"`
	Method      string  `json:"method"` // text | activity
	AnchorsUsed int     `json:"anchors_used"`
	ResidualMs  float64 `json:"residual_ms"`
	Confidence  float64 `json:"confidence"`
}

func (m *Mapping) PieceFor(t float64) Piece {
	for _, p := range m.Pieces {
		if t >= p.Start && t < p.End {
			return p
		}
	}
	if t < m.Pieces[0].Start {
		return m.Pieces[0]
	}
	return m.Pieces[len(m.Pieces)-1]
}

func (m *Mapping) scale() float64 {
	if m.Scale == 0 {
		return 1
	}
	return m.Scale
}

// ShiftAt is how far a cue starting at t (original subtitle time) moves.
func (m *Mapping) ShiftAt(t float64) float64 {
	st := t * m.scale()
	return st + m.PieceFor(st).Shift(st) - t
}

// Apply maps a cue's start/end; both use the piece chosen by the start so a cue never straddles a cut.
func (m *Mapping) Apply(start, end float64) (float64, float64) {
	k := m.scale()
	s, e := start*k, end*k
	p := m.PieceFor(s)
	return s + p.Shift(s), e + p.Shift(e)
}

func scaleCues(cues []Cue, k float64) []Cue {
	out := make([]Cue, len(cues))
	for i, c := range cues {
		out[i] = Cue{Start: c.Start * k, End: c.End * k, Tokens: c.Tokens}
	}
	return out
}

// Common framerate mix-ups (subtitle made for one rate, video at another).
var frameRatios = []float64{
	25.0 / 23.976, 23.976 / 25.0, 25.0 / 24.0, 24.0 / 25.0, 24.0 / 23.976, 23.976 / 24.0,
	30.0 / 25.0, 25.0 / 30.0, 29.97 / 23.976, 23.976 / 29.97,
}

type Params struct {
	MaxOffset      float64
	MatchThreshold float64
	MinAnchors     int
	SyncThreshold  float64 // seconds
	SplitTolerance float64
	MinSegment     int
	AllowActivity  bool
}

func DefaultParams() Params {
	return Params{MaxOffset: 120, MatchThreshold: 80, MinAnchors: 8, SyncThreshold: 0.3,
		SplitTolerance: 0.6, MinSegment: 5, AllowActivity: true}
}

type Result struct {
	Status        string   `json:"status"` // in_sync | needs_fix | uncertain
	Message       string   `json:"message"`
	Mapping       *Mapping `json:"mapping,omitempty"`
	Cues          int      `json:"cues"`
	Anchors       int      `json:"anchors"`
	MedianShiftMs int64    `json:"median_shift_ms"`
	MaxShiftMs    int64    `json:"max_shift_ms"`
	WantsFull     bool     `json:"-"`
}

// ---------------------------------------------------------------------------
// Step 1: anchors
// ---------------------------------------------------------------------------

type token struct {
	text     string
	t        float64
	reliable bool
}

func transcriptTokens(words []Word) []token {
	var out []token
	for _, w := range words {
		for i, tk := range Tokenize(w.Text, false) {
			out = append(out, token{tk, w.Start, w.Reliable && i == 0})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].t < out[j].t })
	return out
}

const maxCueWords = 10

// FindAnchors matches cue starts to speech. With reliableOnly, a cue may only anchor on a
// transcript position whose timestamp is trustworthy.
func FindAnchors(cues []Cue, words []Word, p Params, reliableOnly bool) []Anchor {
	toks := transcriptTokens(words)
	if len(toks) == 0 {
		return nil
	}
	var anchors []Anchor
	type cand struct{ s, t float64 }
	for _, c := range cues {
		cw := c.Tokens
		if len(cw) > maxCueWords {
			cw = cw[:maxCueWords]
		}
		phrase := strings.Join(cw, " ")
		if len(phrase) < 8 || (len(cw) < 3 && len(phrase) < 12) {
			continue // too short to match reliably
		}
		first := cw[0]
		lo := sort.Search(len(toks), func(i int) bool { return toks[i].t >= c.Start-p.MaxOffset })
		hi := sort.Search(len(toks), func(i int) bool { return toks[i].t > c.Start+p.MaxOffset })
		var cands []cand
		for i := lo; i < hi; i++ {
			if reliableOnly && !toks[i].reliable {
				continue
			}
			w := toks[i].text
			if w != first && (len(first) < 4 || Ratio(w, first) < 75) {
				continue
			}
			end := i + len(cw)
			if end > len(toks) {
				end = len(toks)
			}
			parts := make([]string, 0, end-i)
			for _, tk := range toks[i:end] {
				parts = append(parts, tk.text)
			}
			if s := Ratio(phrase, strings.Join(parts, " ")); s >= p.MatchThreshold-15 {
				cands = append(cands, cand{s, toks[i].t})
			}
		}
		if len(cands) == 0 {
			continue
		}
		sort.Slice(cands, func(i, j int) bool { return cands[i].s > cands[j].s })
		best := cands[0]
		if best.s < p.MatchThreshold {
			continue
		}
		// reject phrases that match about equally well somewhere else ("yeah, yeah, yeah")
		ambiguous := false
		for _, o := range cands[1:] {
			if math.Abs(o.t-best.t) > 1.5 {
				ambiguous = best.s-o.s < 8
				break
			}
		}
		if !ambiguous {
			anchors = append(anchors, Anchor{SubTime: c.Start, AudioTime: best.t, Score: best.s})
		}
	}
	sort.Slice(anchors, func(i, j int) bool { return anchors[i].SubTime < anchors[j].SubTime })
	return anchors
}

// ---------------------------------------------------------------------------
// Step 2: robust fit
// ---------------------------------------------------------------------------

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func deltas(as []Anchor) []float64 {
	out := make([]float64, len(as))
	for i, a := range as {
		out[i] = a.Delta()
	}
	return out
}

func localInliers(as []Anchor, tol float64) []Anchor {
	const k = 3
	var keep []Anchor
	for i, a := range as {
		agree, neigh := 0, 0
		for j := i - k; j <= i+k; j++ {
			if j < 0 || j >= len(as) || j == i {
				continue
			}
			neigh++
			if math.Abs(as[j].Delta()-a.Delta()) <= tol {
				agree++
			}
		}
		need := 2
		if neigh < need {
			need = neigh
		}
		if agree >= need {
			keep = append(keep, a)
		}
	}
	return keep
}

func tail(as []Anchor, n int) []Anchor {
	if len(as) > n {
		return as[len(as)-n:]
	}
	return as
}

// predict extrapolates the delta at time t from the most recent anchors (linear if they span enough time).
func predict(as []Anchor, t float64) float64 {
	as = tail(as, 8)
	if len(as) < 4 || as[len(as)-1].SubTime-as[0].SubTime < 30 {
		return median(deltas(tail(as, 5)))
	}
	var mx, my float64
	for _, a := range as {
		mx += a.SubTime
		my += a.Delta()
	}
	n := float64(len(as))
	mx, my = mx/n, my/n
	var sxx, sxy float64
	for _, a := range as {
		sxx += (a.SubTime - mx) * (a.SubTime - mx)
		sxy += (a.SubTime - mx) * (a.Delta() - my)
	}
	slope := sxy / sxx
	if math.Abs(slope) > 0.1 {
		return median(deltas(tail(as, 5)))
	}
	return my + slope*(t-mx)
}

func segment(as []Anchor, tol float64, minSize int) [][]Anchor {
	segs := [][]Anchor{{as[0]}}
	for _, a := range as[1:] {
		cur := segs[len(segs)-1]
		if math.Abs(a.Delta()-predict(cur, a.SubTime)) <= tol {
			segs[len(segs)-1] = append(cur, a)
		} else {
			segs = append(segs, []Anchor{a})
		}
	}
	var big [][]Anchor
	for _, s := range segs {
		if len(s) >= minSize {
			big = append(big, s)
		}
	}
	if len(big) == 0 {
		longest := segs[0]
		for _, s := range segs {
			if len(s) > len(longest) {
				longest = s
			}
		}
		big = [][]Anchor{longest}
	}
	merged := [][]Anchor{big[0]}
	for _, s := range big[1:] {
		last := merged[len(merged)-1]
		if math.Abs(median(deltas(s[:min(5, len(s))]))-predict(last, s[0].SubTime)) <= tol {
			merged[len(merged)-1] = append(last, s...)
		} else {
			merged = append(merged, s)
		}
	}
	return merged
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func fitPiece(seg []Anchor) Piece {
	n := float64(len(seg))
	var mean float64
	for _, a := range seg {
		mean += a.SubTime
	}
	mean /= n
	ds := deltas(seg)
	p := Piece{Offset: median(ds), Ref: mean, Anchors: len(seg)}
	span := seg[len(seg)-1].SubTime - seg[0].SubTime
	if len(seg) >= 8 && span >= 300 {
		var sxx, sxy, sy float64
		for _, a := range seg {
			x := a.SubTime - mean
			sxx += x * x
			sxy += x * a.Delta()
			sy += a.Delta()
		}
		if sxx > 0 {
			slope := sxy / sxx
			icpt := sy / n
			linRes := make([]float64, len(seg))
			constRes := make([]float64, len(seg))
			for i, a := range seg {
				linRes[i] = math.Abs(a.Delta() - (icpt + slope*(a.SubTime-mean)))
				constRes[i] = math.Abs(a.Delta() - p.Offset)
			}
			// accept drift only if it is real (>= 0.2 s across the segment) and clearly fits better
			if math.Abs(slope) < 0.1 && math.Abs(slope)*span >= 0.2 && median(linRes) < median(constRes)*0.8 {
				p.Offset, p.Slope = icpt, slope
			}
		}
	}
	return p
}

// boundary picks where to switch between two segments: the largest gap between cues in (t1, t2].
func boundary(cues []Cue, t1, t2 float64) float64 {
	best, bestGap := t2, -1.0
	prevEnd := t1
	for _, c := range cues {
		if c.Start <= t1 {
			prevEnd = math.Max(prevEnd, c.End)
			continue
		}
		if c.Start > t2 {
			break
		}
		if gap := c.Start - prevEnd; gap > bestGap {
			best, bestGap = c.Start, gap
		}
		prevEnd = math.Max(prevEnd, c.End)
	}
	return best
}

func FitMapping(anchors []Anchor, cues []Cue, p Params) *Mapping {
	if len(anchors) < p.MinAnchors {
		return nil
	}
	inl := localInliers(anchors, p.SplitTolerance)
	if len(inl) < p.MinAnchors {
		return nil
	}
	segs := segment(inl, p.SplitTolerance, p.MinSegment)
	pieces := make([]Piece, len(segs))
	for i, s := range segs {
		pieces[i] = fitPiece(s)
	}
	pieces[0].Start = math.Inf(-1)
	pieces[len(pieces)-1].End = math.Inf(1)
	for i := 0; i < len(segs)-1; i++ {
		b := boundary(cues, segs[i][len(segs[i])-1].SubTime, segs[i+1][0].SubTime)
		pieces[i].End, pieces[i+1].Start = b, b
	}
	m := &Mapping{Scale: 1, Pieces: pieces, Method: "text"}
	var res []float64
	for _, s := range segs {
		for _, a := range s {
			m.AnchorsUsed++
			res = append(res, math.Abs(a.Delta()-m.ShiftAt(a.SubTime)))
		}
	}
	m.ResidualMs = math.Round(median(res) * 1000)
	m.Confidence = math.Round(1000*float64(m.AnchorsUsed)/float64(len(anchors))) / 1000
	return m
}

// ---------------------------------------------------------------------------
// Step 3: language-agnostic fallback
// ---------------------------------------------------------------------------

// CorrelateOffset finds the global shift that best lines "subtitle on screen" up with "speech heard".
func CorrelateOffset(cues []Cue, speech, spans [][2]float64, duration, maxOffset float64) *Mapping {
	const res = 0.1
	if len(cues) == 0 || len(speech) == 0 || duration <= 0 {
		return nil
	}
	n := int(math.Ceil((duration+maxOffset)/res)) + 1
	idx := func(t float64) int {
		i := int(math.Round(t / res))
		if i < 0 {
			return 0
		}
		if i > n {
			return n
		}
		return i
	}
	a := make([]float32, n) // audio: +1 speech, -1 silence, 0 not analysed
	for _, s := range spans {
		for i := idx(s[0]); i < idx(s[1]); i++ {
			a[i] = -1
		}
	}
	for _, s := range speech {
		for i := idx(s[0]); i < idx(s[1]); i++ {
			if a[i] != 0 {
				a[i] = 1
			}
		}
	}
	b := make([]float32, n) // subtitles: +1 on screen, -1 otherwise
	for i := range b {
		b[i] = -1
	}
	for _, c := range cues {
		for i := idx(c.Start); i < idx(c.End); i++ {
			b[i] = 1
		}
	}
	var active []int
	for i, v := range a {
		if v != 0 {
			active = append(active, i)
		}
	}
	K := int(maxOffset / res)
	vals := make([]float64, 2*K+1)
	for k := -K; k <= K; k++ { // sum_t a[t] * b[t-k]  ->  delta = k*res
		var s float64
		for _, t := range active {
			if j := t - k; j >= 0 && j < n {
				s += float64(a[t] * b[j])
			}
		}
		vals[k+K] = s
	}
	best := 0
	var mean float64
	for i, v := range vals {
		mean += v
		if v > vals[best] {
			best = i
		}
	}
	mean /= float64(len(vals))
	var variance float64
	for _, v := range vals {
		variance += (v - mean) * (v - mean)
	}
	std := math.Sqrt(variance / float64(len(vals)))
	if std == 0 {
		return nil
	}
	rival := mean
	for i, v := range vals {
		if abs(i-best) > int(1.0/res) && v > rival {
			rival = v
		}
	}
	z, sharp := (vals[best]-mean)/std, (vals[best]-rival)/std
	if z < 4 || sharp < 0.75 {
		return nil
	}
	off := float64(best-K) * res
	return &Mapping{Scale: 1, Pieces: []Piece{{Start: math.Inf(-1), End: math.Inf(1), Offset: off}},
		Method: "activity", Confidence: math.Round(math.Min(1, sharp/3)*1000) / 1000}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

func decide(cues []Cue, m *Mapping, p Params, r *Result) {
	shifts := make([]float64, 0, len(cues))
	maxAbs := 0.0
	for _, c := range cues {
		s := m.ShiftAt(c.Start)
		shifts = append(shifts, s)
		maxAbs = math.Max(maxAbs, math.Abs(s))
	}
	r.Mapping = m
	r.MedianShiftMs = int64(math.Round(median(shifts) * 1000))
	r.MaxShiftMs = int64(math.Round(maxAbs * 1000))
	if maxAbs < p.SyncThreshold {
		r.Status = "in_sync"
		r.Message = "In sync (max deviation " + itoa(r.MaxShiftMs) + " ms)"
		return
	}
	r.Status = "needs_fix"
	if len(m.Pieces) == 1 {
		pc := m.Pieces[0]
		r.Message = "Shift " + signed(pc.Offset*1000) + " ms"
		if pc.Slope != 0 {
			r.Message += ", drift " + signedF(pc.Slope*3600) + " s/h"
		}
	} else {
		parts := make([]string, len(m.Pieces))
		for i, pc := range m.Pieces {
			parts[i] = signed(pc.Offset * 1000)
		}
		r.Message = "Shifts " + strings.Join(parts, ", ") + " ms across " + itoa(int64(len(m.Pieces))) + " segments"
	}
}

func fitScaled(cues []Cue, words []Word, p Params, k float64) (*Mapping, int) {
	sc := cues
	if k != 1 {
		sc = scaleCues(cues, k)
	}
	anchors := FindAnchors(sc, words, p, true)
	m := FitMapping(anchors, sc, p)
	if m == nil { // not enough trustworthy positions: use every word (noisier, outliers get filtered)
		anchors = FindAnchors(sc, words, p, false)
		m = FitMapping(anchors, sc, p)
	}
	if m != nil {
		m.Scale = k
	}
	return m, len(anchors)
}

func needsRatioSearch(m *Mapping, cues int) bool {
	if m == nil {
		return true
	}
	if len(m.Pieces) > 2 || float64(m.AnchorsUsed) < 0.25*float64(cues) {
		return true
	}
	for _, pc := range m.Pieces {
		if math.Abs(pc.Slope) > 0.01 {
			return true
		}
	}
	return false
}

// Analyze decides whether a subtitle track is in sync and how to fix it.
func Analyze(cues []Cue, words []Word, speech, spans [][2]float64, duration float64, p Params, sampled bool) Result {
	r := Result{Cues: len(cues)}
	m, nAnchors := fitScaled(cues, words, p, 1)
	if needsRatioSearch(m, len(cues)) && len(words) > 0 {
		for _, k := range frameRatios {
			m2, n2 := fitScaled(cues, words, p, k)
			if m2 == nil {
				continue
			}
			better := m == nil ||
				float64(m2.AnchorsUsed) > 1.2*float64(m.AnchorsUsed) ||
				(m2.AnchorsUsed >= m.AnchorsUsed && len(m2.Pieces) < len(m.Pieces))
			if better {
				m, nAnchors = m2, n2
			}
		}
	}
	r.Anchors = nAnchors
	if m != nil {
		decide(cues, m, p, &r)
		if m.Scale != 1 {
			r.Message += " (framerate " + strconv.FormatFloat(m.Scale, 'f', 4, 64) + "x)"
		}
		r.WantsFull = sampled && len(m.Pieces) > 1 // sampled windows can't place cut points precisely
		return r
	}
	if p.AllowActivity {
		if m := CorrelateOffset(cues, speech, spans, duration, p.MaxOffset); m != nil {
			decide(cues, m, p, &r)
			r.Message += " (low confidence: subtitle text did not match the speech)"
			r.WantsFull = sampled
			return r
		}
	}
	r.Status = "uncertain"
	r.Message = "Could not determine timing (" + itoa(int64(nAnchors)) + " text matches)"
	r.WantsFull = sampled
	return r
}
