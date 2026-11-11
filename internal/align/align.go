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
	// Contradicted: a run of consecutive matches (or a whole sample window) consistently disagrees
	// with this mapping - e.g. a cold open or tail with its own offset that the fit absorbed.
	Contradicted bool `json:"contradicted,omitempty"`
	Disagreeing  int  `json:"disagreeing,omitempty"` // matches in contradicting runs/windows

	agree    []float64 // audio times of anchors that agree with this mapping (coverage check)
	disagree []float64 // audio times of plausible anchors (within 5 s) that don't
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

func (m *Mapping) disagreeIn(w [2]float64) int {
	n := 0
	for _, t := range m.disagree {
		if t >= w[0] && t < w[1] {
			n++
		}
	}
	return n
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
	Status        string   `json:"status"`           // in_sync | needs_fix | uncertain
	Reason        string   `json:"reason,omitempty"` // why uncertain: few_matches | parts_disagree | no_match | activity_ambiguous
	Message       string   `json:"message"`
	Mapping       *Mapping `json:"mapping,omitempty"`
	Cues          int      `json:"cues"`
	Anchors       int      `json:"anchors"`
	MedianShiftMs int64    `json:"median_shift_ms"`
	MaxShiftMs    int64    `json:"max_shift_ms"`
	WantsFull     bool     `json:"-"`
	// SampleDisagrees: part of the sampled audio contradicted the result (not just thin evidence) -
	// such a result must not stand as "in sync" without a full pass.
	SampleDisagrees bool `json:"-"`
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

// maxLineDrift is the largest drift (s per s, ~10.8 s/h) fitted as a line; bigger framerate
// mismatches are handled by the framerate search, which rescales the subtitle first.
const maxLineDrift = 0.003

// fitLine robustly fits delta = offset + slope*(t-ref) and returns the anchors it explains.
// Drift is only kept when it is statistically clear and makes a real difference, so noisy
// anchors never produce a fake slope.
func fitLine(as []Anchor, tol float64) (Piece, []Anchor) {
	n := len(as)
	var ref float64
	for _, a := range as {
		ref += a.SubTime
	}
	ref /= float64(n)
	span := as[n-1].SubTime - as[0].SubTime

	slope := 0.0
	if n >= 10 && span >= 600 { // Theil-Sen over pairs far apart: robust to wrong matches
		var slopes []float64
		stride := 1 + n/80
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j += stride {
				if dt := as[j].SubTime - as[i].SubTime; dt >= 300 {
					slopes = append(slopes, (as[j].Delta()-as[i].Delta())/dt)
				}
			}
		}
		if len(slopes) >= 20 {
			slope = median(slopes)
		}
		if math.Abs(slope) > maxLineDrift {
			slope = 0
		}
	}
	off := func(sl float64, set []Anchor) float64 {
		r := make([]float64, len(set))
		for i, a := range set {
			r[i] = a.Delta() - sl*(a.SubTime-ref)
		}
		return median(r)
	}
	inliers := func(o, sl float64) []Anchor {
		var in []Anchor
		for _, a := range as {
			if math.Abs(a.Delta()-(o+sl*(a.SubTime-ref))) <= tol {
				in = append(in, a)
			}
		}
		return in
	}
	o := off(slope, as)
	in := inliers(o, slope)
	if slope != 0 && len(in) >= 10 {
		// least squares on the inliers, then keep the slope only if it is significant
		var mx, my float64
		for _, a := range in {
			mx += a.SubTime
			my += a.Delta()
		}
		k := float64(len(in))
		mx, my = mx/k, my/k
		var sxx, sxy float64
		for _, a := range in {
			sxx += (a.SubTime - mx) * (a.SubTime - mx)
			sxy += (a.SubTime - mx) * (a.Delta() - my)
		}
		ls := sxy / sxx
		var rss float64
		for _, a := range in {
			e := a.Delta() - (my + ls*(a.SubTime-mx))
			rss += e * e
		}
		se := math.Sqrt(rss/math.Max(1, k-2)) / math.Sqrt(sxx)
		inSpan := in[len(in)-1].SubTime - in[0].SubTime
		if math.Abs(ls) <= maxLineDrift && math.Abs(ls) >= 4*se && math.Abs(ls)*inSpan >= 0.25 {
			slope = ls
		} else {
			slope = 0
		}
	} else {
		slope = 0
	}
	o = off(slope, in)
	if len(in) == 0 {
		o = off(slope, as)
	}
	in = inliers(o, slope)
	if len(in) > 0 {
		o = off(slope, in)
	}
	return Piece{Offset: o, Slope: slope, Ref: ref, Anchors: len(in)}, in
}

// mergeSegments joins neighbouring segments that are really one: timing is continuous where
// they meet (a jump is a discontinuity; drift is not) and one line explains them both.
func mergeSegments(segs [][]Anchor, tol float64) [][]Anchor {
	for changed := true; changed && len(segs) > 1; {
		changed = false
		for i := 0; i < len(segs)-1; i++ {
			end := median(deltas(tail(segs[i], 6)))
			start := median(deltas(segs[i+1][:min(6, len(segs[i+1]))]))
			if math.Abs(end-start) > tol {
				continue
			}
			union := append(append([]Anchor(nil), segs[i]...), segs[i+1]...)
			if _, in := fitLine(union, tol); float64(len(in)) >= 0.85*float64(len(union)) {
				segs = append(segs[:i], append([][]Anchor{union}, segs[i+2:]...)...)
				changed = true
				break
			}
		}
	}
	return segs
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

// placeSwitch picks where timing changes from piece a to piece b, somewhere in (t1, t2].
// Every loose match in that stretch votes for the piece whose shift it fits better; the switch
// goes where the fewest votes disagree. Without votes it falls back to the largest gap.
//
// Loose matches are typically biased (e.g. late), so the bias is measured on each side, where
// the right piece is already known, and removed before voting.
func placeSwitch(cues []Cue, loose []Anchor, a, b Piece, aFrom, t1, t2, bTo float64) float64 {
	fallback := boundary(cues, t1, t2)
	bias := func(pc Piece, from, to float64) (float64, bool) {
		var r []float64
		for _, x := range loose {
			if x.SubTime >= from && x.SubTime <= to {
				if e := x.Delta() - pc.Shift(x.SubTime); math.Abs(e) < 5 {
					r = append(r, e)
				}
			}
		}
		return median(r), len(r) >= 5
	}
	biasA, okA := bias(a, math.Max(aFrom, t1-900), t1)
	biasB, okB := bias(b, t2, math.Min(bTo, t2+900))
	if !okA || !okB {
		return fallback
	}
	var votes []Anchor
	for _, x := range loose {
		if x.SubTime > t1 && x.SubTime < t2 {
			votes = append(votes, x)
		}
	}
	if len(votes) < 3 {
		return fallback
	}
	cost := func(sw float64) float64 {
		var c float64
		for _, v := range votes {
			e := v.Delta() - a.Shift(v.SubTime) - biasA
			if v.SubTime >= sw {
				e = v.Delta() - b.Shift(v.SubTime) - biasB
			}
			c += math.Min(math.Abs(e), 2)
		}
		return c
	}
	best, bestCost := fallback, cost(fallback)
	for _, c := range cues {
		if c.Start <= t1 || c.Start > t2 {
			continue
		}
		if k := cost(c.Start); k < bestCost-1e-9 {
			best, bestCost = c.Start, k
		}
	}
	return best
}

// dropExcursions removes middle sections whose neighbours agree with each other (A-B-A). A stretch
// at another offset that returns to the same timing is far more likely a biased stretch of
// transcript than a real edit; its anchors are dropped, then flagged as contradicting (-> uncertain).
func dropExcursions(segs [][]Anchor, tol float64) [][]Anchor {
	for changed := true; changed && len(segs) > 2; {
		changed = false
		for i := 1; i < len(segs)-1; i++ {
			mid := segs[i]
			span := mid[len(mid)-1].SubTime - mid[0].SubTime
			before := median(deltas(tail(segs[i-1], 6)))
			after := median(deltas(segs[i+1][:min(6, len(segs[i+1]))]))
			_ = span
			if math.Abs(before-after) <= tol {
				joined := append(append([]Anchor(nil), segs[i-1]...), segs[i+1]...)
				segs = append(segs[:i-1], append([][]Anchor{joined}, segs[i+2:]...)...)
				changed = true
				break
			}
		}
	}
	return segs
}

// sections fits one piece per segment; returns nil when there is only one segment.
// minEdgeSection: a section at the very start or end (cold open, tail) needs more support than one
// in the middle - from matches alone it can't be told apart from a stretch of biased transcript
// (music/titles), so thin edge sections are dropped and then flag the result as contradicted.
const minEdgeSection = 12

func sections(inl, loose []Anchor, cues []Cue, tol float64, minSeg int) ([]Piece, []Anchor) {
	segs := dropExcursions(mergeSegments(segment(inl, tol, minSeg), tol), tol)
	for len(segs) > 1 && len(segs[0]) < minEdgeSection {
		segs = segs[1:]
	}
	for len(segs) > 1 && len(segs[len(segs)-1]) < minEdgeSection {
		segs = segs[:len(segs)-1]
	}
	if len(segs) < 2 {
		return nil, nil
	}
	var sp []Piece
	var su []Anchor
	for _, sg := range segs {
		pc, in := fitLine(sg, tol)
		sp = append(sp, pc)
		su = append(su, in...)
	}
	sp[0].Start = math.Inf(-1)
	sp[len(sp)-1].End = math.Inf(1)
	for i := 0; i < len(segs)-1; i++ {
		b := placeSwitch(cues, loose, sp[i], sp[i+1], segs[i][0].SubTime,
			segs[i][len(segs[i])-1].SubTime, segs[i+1][0].SubTime, segs[i+1][len(segs[i+1])-1].SubTime)
		sp[i].End, sp[i+1].Start = b, b
	}
	return sp, su
}

// consistentRuns looks at ALL anchors (before outlier filtering, which can remove a sparse block):
// runs of 3+ anchors in time order that disagree with the mapping in the same direction AND agree
// with each other (spread <= 2*tol). Wild matches (> 5 s off) are skipped rather than breaking a
// run; an anchor that agrees with the mapping ends it. Random wrong matches scatter by tens of
// seconds, so they almost never form such a run.
func consistentRuns(all []Anchor, m *Mapping, tol float64) int {
	bad := 0
	var run []float64
	flush := func() {
		if len(run) >= 3 {
			bad += len(run)
		}
		run = run[:0]
	}
	for _, a := range all {
		r := a.Delta() - m.ShiftAt(a.SubTime)
		switch {
		case math.Abs(r) <= tol:
			flush()
		case math.Abs(r) > 5:
			// wild match: ignore
		default:
			if len(run) > 0 {
				lo, hi := r, r
				for _, x := range run {
					lo, hi = math.Min(lo, x), math.Max(hi, x)
				}
				if (r > 0) != (run[0] > 0) || hi-lo > 2*tol {
					flush()
				}
			}
			run = append(run, r)
		}
	}
	flush()
	return bad
}

// contradictions counts matches that consistently disagree with the mapping: runs of 3+
// consecutive anchors off by more than tol in the same direction, and sample windows (spans, in
// audio time) whose anchors disagree on median.
func contradictions(inl []Anchor, m *Mapping, spans [][2]float64, tol float64) int {
	resid := make([]float64, len(inl))
	for i, a := range inl {
		resid[i] = a.Delta() - m.ShiftAt(a.SubTime)
	}
	bad := 0
	for i := 0; i < len(resid); {
		if math.Abs(resid[i]) <= tol {
			i++
			continue
		}
		j := i
		for j < len(resid) && math.Abs(resid[j]) > tol && (resid[j] > 0) == (resid[i] > 0) {
			j++
		}
		if j-i >= 3 {
			bad += j - i
		}
		i = j
	}
	if len(spans) > 1 {
		for _, w := range spans {
			var r []float64
			for i, a := range inl {
				if a.AudioTime >= w[0] && a.AudioTime < w[1] {
					r = append(r, resid[i])
				}
			}
			if len(r) >= 2 && math.Abs(median(r)) > tol {
				bad += len(r)
			}
		}
	}
	return bad
}

// FitMapping chooses between one line for the whole file (offset, plus drift if it is real)
// and separate sections (when the timing genuinely jumps, e.g. a different cut). Both models are
// always evaluated - a cold open or tail with its own offset is a minority a single line absorbs.
// The chosen mapping is then checked for blocks of matches that still disagree with it.
//
// loose are lower-precision matches (any word, not just segment starts); they are only used to
// decide where exactly a jump happens. spans are the analysed audio windows. Both may be nil.
func FitMapping(anchors, loose []Anchor, cues []Cue, spans [][2]float64, p Params) *Mapping {
	need := p.MinAnchors
	if need < 3 {
		need = 3
	}
	if len(anchors) < need {
		return nil
	}
	tol := p.SplitTolerance
	inl := localInliers(anchors, tol)
	if len(inl) < need {
		return nil
	}

	line, lineIn := fitLine(inl, tol)
	line.Start, line.End = math.Inf(-1), math.Inf(1)
	pieces, used := []Piece{line}, lineIn

	minSeg := p.MinSegment
	if minSeg < 8 {
		minSeg = 8
	}
	// Sections that survive merging meet at a verified jump; they win when they explain at least as
	// much as the single line (a small per-section cost breaks ties).
	if sp, su := sections(inl, loose, cues, tol, minSeg); sp != nil && len(su)-2*(len(sp)-1) >= len(lineIn) {
		pieces, used = sp, su
	}

	build := func(pcs []Piece, u []Anchor) *Mapping {
		m := &Mapping{Scale: 1, Pieces: pcs, Method: "text", AnchorsUsed: len(u)}
		res := make([]float64, 0, len(u))
		for _, a := range u {
			res = append(res, math.Abs(a.Delta()-m.ShiftAt(a.SubTime)))
		}
		m.ResidualMs = math.Round(median(res) * 1000)
		m.Confidence = math.Round(1000*float64(m.AnchorsUsed)/float64(len(anchors))) / 1000
		m.Disagreeing = contradictions(inl, m, spans, tol)
		if c := consistentRuns(anchors, m, tol); c > m.Disagreeing {
			m.Disagreeing = c
		}
		for _, a := range anchors {
			switch r := math.Abs(a.Delta() - m.ShiftAt(a.SubTime)); {
			case r <= tol:
				m.agree = append(m.agree, a.AudioTime)
			case r <= 5:
				m.disagree = append(m.disagree, a.AudioTime)
			}
		}
		sort.Float64s(m.agree)
		m.Contradicted = m.Disagreeing > 0
		return m
	}
	// A block too small to stand as its own section (cold open, tail) but contradicting the fit
	// makes the result untrustworthy (-> uncertain): we never rewrite it on thin evidence.
	return build(pieces, used)
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
			bj := float32(-1) // before/after the film: no subtitle on screen (skipping it biased the peak)
			if j := t - k; j >= 0 && j < n {
				bj = b[j]
			}
			s += float64(a[t] * bj)
		}
		vals[k+K] = s
	}
	best := 0
	var mean float64
	for i, v := range vals {
		mean += v
		// on a plateau of equal scores take the most positive lag: the plateau extends from the
		// true lag towards earlier lags (cue starts must line up with speech onsets)
		if v >= vals[best] {
			best = i
		}
	}
	plateau := 0
	for i := best; i >= 0 && vals[i] == vals[best]; i-- {
		plateau++
	}
	if float64(plateau)*res > 1.0 { // cues outlast speech a little, so short plateaus are normal
		return nil // too ambiguous to act on
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

// activityAgrees runs the activity correlation separately on six parts of the film. Each part with
// enough subtitles must find a clear peak within 0.5 s of the global offset - otherwise the film's
// timing isn't one global shift (a jump, drift) and the global answer can't be trusted.
func activityAgrees(cues []Cue, speech, spans [][2]float64, duration, maxOffset, global float64) bool {
	const parts = 6
	for i := 0; i < parts; i++ {
		lo, hi := duration*float64(i)/parts, duration*float64(i+1)/parts
		if i == parts-1 {
			hi = math.Inf(1)
		}
		var c []Cue
		for _, x := range cues {
			if x.Start >= lo && x.Start < hi {
				c = append(c, x)
			}
		}
		if len(c) < 20 {
			continue
		}
		var sp, sn [][2]float64
		for _, x := range speech {
			if x[0] >= lo-maxOffset && x[0] < hi+maxOffset {
				sp = append(sp, x)
			}
		}
		for _, w := range spans {
			a, b := math.Max(w[0], lo-maxOffset), math.Min(w[1], hi+maxOffset)
			if b > a {
				sn = append(sn, [2]float64{a, b})
			}
		}
		m := CorrelateOffset(c, sp, sn, duration, maxOffset)
		if m == nil || math.Abs(m.Pieces[0].Offset-global) > 0.5 {
			return false
		}
	}
	return true
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

func fitScaled(cues []Cue, words []Word, spans [][2]float64, p Params, k float64) (*Mapping, int) {
	sc := cues
	if k != 1 {
		sc = scaleCues(cues, k)
	}
	anchors := FindAnchors(sc, words, p, true)
	loose := FindAnchors(sc, words, p, false)
	m := FitMapping(anchors, loose, sc, spans, p)
	if m == nil { // not enough trustworthy positions: use every word (noisier, outliers get filtered)
		anchors = loose
		m = FitMapping(anchors, loose, sc, spans, p)
	}
	if m != nil {
		m.Scale = k
	}
	return m, len(anchors)
}

// needsRatioSearch: try framerate rescaling only when the plain fit is missing, split up or
// explains too few of its anchors (a big framerate mismatch looks like that).
func needsRatioSearch(m *Mapping) bool {
	return m == nil || len(m.Pieces) > 1 || m.Confidence < 0.6 || m.Contradicted
}

// cleanSample: a result from sampled windows is only trusted when the evidence is simple and
// consistent - one offset, no drift, little disagreement - and the parts of the film that were
// heard back it: no window with speech whose plausible matches mostly disagree, and no run of 3+
// windows with speech (about a quarter of the film) without an agreeing match. Windows without
// speech (music, credits) don't count. disagrees reports actual contradicting evidence (as opposed
// to merely thin evidence).
func cleanSample(m *Mapping, words []Word, spans [][2]float64, tol float64) (clean, disagrees bool) {
	if m.Contradicted {
		return false, true
	}
	if len(m.Pieces) != 1 || m.Pieces[0].Slope != 0 || m.Confidence < 0.75 || len(m.agree) == 0 {
		return false, false
	}
	missed := 0
	for _, w := range spans {
		nw := 0
		for _, x := range words {
			if x.Start >= w[0] && x.Start < w[1] {
				nw++
			}
		}
		if nw < 5 {
			continue // no real speech here
		}
		agree := 0
		for _, t := range m.agree {
			if t >= w[0] && t < w[1] {
				agree++
			}
		}
		if dis := m.disagreeIn(w); dis >= 2 && dis > agree {
			return false, true // this part of the film contradicts the result
		}
		if agree > 0 {
			missed = 0
			continue
		}
		if missed++; missed >= 3 {
			return false, false
		}
	}
	return true, false
}

// minConfidentAnchors is the least evidence we act on, whatever MIN_ANCHORS says.
const minConfidentAnchors = 12

// weak reports whether a mapping rests on too little or too contradictory evidence to act on.
func weak(m *Mapping, p Params) bool {
	need := p.MinAnchors
	if need < minConfidentAnchors {
		need = minConfidentAnchors
	}
	return m.AnchorsUsed < need || m.Confidence < 0.5
}

// Analyze decides whether a subtitle track is in sync and how to fix it.
func Analyze(cues []Cue, words []Word, speech, spans [][2]float64, duration float64, p Params, sampled bool) Result {
	r := Result{Cues: len(cues)}
	m, nAnchors := fitScaled(cues, words, spans, p, 1)
	if needsRatioSearch(m) && len(words) > 0 {
		base := m
		for _, k := range frameRatios {
			m2, n2 := fitScaled(cues, words, spans, p, k)
			if m2 == nil || weak(m2, p) || m2.Confidence < 0.6 || m2.Contradicted {
				continue
			}
			// a rescale must be clearly better than no rescale, never just different
			better := base == nil ||
				(float64(m2.AnchorsUsed) >= 1.3*float64(base.AnchorsUsed) && len(m2.Pieces) <= len(base.Pieces))
			if better && (m == base || m2.AnchorsUsed > m.AnchorsUsed) {
				m, nAnchors = m2, n2
			}
		}
	}
	r.Anchors = nAnchors
	if m != nil && weak(m, p) {
		r.Status, r.Reason = "uncertain", "few_matches"
		r.Mapping = m
		r.Message = "Not enough reliable matches to act on (" + itoa(int64(m.AnchorsUsed)) + " consistent of " +
			itoa(int64(nAnchors)) + " matched lines), left unchanged"
		r.WantsFull = sampled
		return r
	}
	if m != nil && m.Contradicted {
		r.Status, r.Reason = "uncertain", "parts_disagree"
		r.Mapping = m
		r.Message = "Timing differs between parts of the file (" + itoa(int64(m.Disagreeing)) +
			" matched lines disagree with the best fit), left unchanged"
		r.WantsFull = sampled
		return r
	}
	if m != nil {
		decide(cues, m, p, &r)
		if m.Scale != 1 {
			r.Message += " (framerate " + strconv.FormatFloat(m.Scale, 'f', 4, 64) + "x)"
		}
		// sampled windows can miss a jump, a tail or a cold open: only a clean result stands
		if sampled {
			clean, dis := cleanSample(m, words, spans, p.SplitTolerance)
			r.WantsFull, r.SampleDisagrees = !clean, dis
		}
		return r
	}
	if p.AllowActivity {
		if m := CorrelateOffset(cues, speech, spans, duration, p.MaxOffset); m != nil {
			// one global offset only: refuse unless every part of the film agrees with it
			if !activityAgrees(cues, speech, spans, duration, p.MaxOffset, m.Pieces[0].Offset) {
				r.Status, r.Reason = "uncertain", "activity_ambiguous"
				r.Message = "Subtitle text did not match the speech, and speech activity suggests different " +
					"offsets in different parts of the film; left unchanged"
				r.WantsFull = sampled
				return r
			}
			decide(cues, m, p, &r)
			r.Message += " (low confidence: subtitle text did not match the speech)"
			r.WantsFull = sampled
			return r
		}
	}
	r.Status, r.Reason = "uncertain", "no_match"
	r.Message = "Could not determine timing (" + itoa(int64(nAnchors)) + " text matches)"
	r.WantsFull = sampled
	return r
}
