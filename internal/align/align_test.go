package align

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

var vocab = []string{"where", "did", "you", "put", "the", "keys", "i", "told", "her", "about", "tomorrow",
	"night", "we", "should", "leave", "before", "they", "notice", "dinner", "is", "ready", "come", "downstairs",
	"have", "seen", "my", "brother", "anywhere", "this", "morning", "nobody", "knows", "what", "happened",
	"there", "station", "train", "late", "again", "honestly", "believe", "promise", "remember", "summer"}

// synth builds n cues (true speech times) and a matching transcript with timing jitter.
func synth(n int, seed int64) ([]Cue, []Word) {
	r := rand.New(rand.NewSource(seed))
	var cues []Cue
	var words []Word
	t := 5.0
	for i := 0; i < n; i++ {
		k := 4 + r.Intn(5)
		toks := make([]string, k)
		for j := range toks {
			toks[j] = vocab[r.Intn(len(vocab))]
		}
		start := t
		wt := start
		for _, tk := range toks {
			jit := (r.Float64() - 0.5) * 0.1
			words = append(words, Word{Text: " " + tk, Start: wt + jit, End: wt + 0.3})
			wt += 0.32
		}
		cues = append(cues, Cue{Start: start, End: wt + 0.4, Tokens: toks})
		t = wt + 1.5 + r.Float64()*4
	}
	return cues, words
}

func shift(cues []Cue, f func(t float64) float64) []Cue {
	out := make([]Cue, len(cues))
	for i, c := range cues {
		out[i] = Cue{Start: f(c.Start), End: f(c.End), Tokens: c.Tokens}
	}
	return out
}

func analyze(cues []Cue, words []Word) Result {
	dur := cues[len(cues)-1].End + 10
	return Analyze(cues, words, nil, [][2]float64{{0, dur}}, dur, DefaultParams(), false)
}

func TestInSync(t *testing.T) {
	cues, words := synth(400, 1)
	r := analyze(shift(cues, func(t float64) float64 { return t + 0.05 }), words)
	if r.Status != "in_sync" {
		t.Fatalf("expected in_sync, got %s: %s", r.Status, r.Message)
	}
}

func TestConstantOffset(t *testing.T) {
	for _, off := range []float64{-7.3, -1.2, 0.8, 2.5, 45} {
		cues, words := synth(400, 2)
		subs := shift(cues, func(t float64) float64 { return t + off })
		r := analyze(subs, words)
		if r.Status != "needs_fix" || len(r.Mapping.Pieces) != 1 {
			t.Fatalf("off %v: got %s %s", off, r.Status, r.Message)
		}
		if got := r.Mapping.Pieces[0].Offset; math.Abs(got+off) > 0.1 {
			t.Fatalf("off %v: estimated %.3f", off, got)
		}
	}
}

func TestFramerateDrift(t *testing.T) {
	cues, words := synth(800, 3)
	ratio := 25.0 / 23.976
	subs := shift(cues, func(t float64) float64 { return t * ratio })
	r := analyze(subs, words)
	if r.Status != "needs_fix" {
		t.Fatalf("got %s", r.Status)
	}
	for _, c := range subs[50:60] {
		s, _ := r.Mapping.Apply(c.Start, c.End)
		if want := c.Start / ratio; math.Abs(s-want) > 0.25 {
			t.Fatalf("drift fix off: %.2f vs %.2f (%s)", s, want, r.Message)
		}
	}
}

func TestSegmentedOffsets(t *testing.T) {
	cues, words := synth(600, 4)
	cut := cues[300].Start
	subs := shift(cues, func(t float64) float64 {
		if t >= cut {
			return t + 3
		}
		return t
	})
	r := analyze(subs, words)
	if r.Status != "needs_fix" || len(r.Mapping.Pieces) != 2 {
		t.Fatalf("expected 2 pieces, got %s %s", r.Status, r.Message)
	}
	bad := 0
	for i, c := range subs {
		s, _ := r.Mapping.Apply(c.Start, c.End)
		if math.Abs(s-cues[i].Start) > 0.25 {
			bad++
		}
	}
	if bad > 3 {
		t.Fatalf("%d cues mis-mapped", bad)
	}
}

func TestOutliersIgnored(t *testing.T) {
	cues, words := synth(400, 5)
	// whisper hallucinating random repeats of real phrases elsewhere
	r0 := rand.New(rand.NewSource(9))
	for i := 0; i < 40; i++ {
		c := cues[r0.Intn(len(cues))]
		base := r0.Float64() * cues[len(cues)-1].End
		for j, tk := range c.Tokens {
			words = append(words, Word{Text: tk, Start: base + float64(j)*0.3})
		}
	}
	r := analyze(shift(cues, func(t float64) float64 { return t - 1.5 }), words)
	if r.Status != "needs_fix" || math.Abs(r.Mapping.Pieces[0].Offset-1.5) > 0.1 {
		t.Fatalf("got %s %s", r.Status, r.Message)
	}
}

func TestActivityFallback(t *testing.T) {
	cues, words := synth(300, 6)
	var speech [][2]float64
	for _, w := range words {
		speech = append(speech, [2]float64{w.Start, w.Start + 0.3})
	}
	foreign := shift(cues, func(t float64) float64 { return t + 4 })
	for i := range foreign {
		foreign[i].Tokens = []string{"lorem", "ipsum", "dolor", fmt.Sprint(i)}
	}
	dur := cues[len(cues)-1].End + 10
	r := Analyze(foreign, nil, speech, [][2]float64{{0, dur}}, dur, DefaultParams(), false)
	if r.Status != "needs_fix" || r.Mapping.Method != "activity" || math.Abs(r.Mapping.Pieces[0].Offset+4) > 0.3 {
		t.Fatalf("got %s %+v", r.Message, r.Mapping)
	}
}

func TestTokenize(t *testing.T) {
	got := Tokenize("JOHN: [door slams] <i>Don’t</i> go, Café!\\N{\\an8}(sighs) ♪ la la ♪", true)
	want := []string{"don't", "go", "cafe"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %v", got)
	}
	if Ratio("hello world", "hello world") != 100 || Ratio("abc", "xyz") != 0 {
		t.Fatal("ratio")
	}
}

// whisper.cpp-like transcript: segment starts are accurate, words inside a segment are smeared
// across it (late), and some segments span two subtitle cues.
func TestSegmentStartAnchors(t *testing.T) {
	cues, _ := synth(300, 11)
	r := rand.New(rand.NewSource(3))
	var words []Word
	for i := 0; i < len(cues); i++ {
		group := []Cue{cues[i]}
		if r.Float64() < 0.35 && i+1 < len(cues) {
			group = append(group, cues[i+1])
			i++
		}
		segStart := group[0].Start
		segEnd := group[len(group)-1].End + 2.5 // trailing silence makes inner words drift late
		var toks []string
		for _, c := range group {
			toks = append(toks, c.Tokens...)
		}
		step := (segEnd - segStart) / float64(len(toks))
		for j, tk := range toks {
			words = append(words, Word{Text: tk, Start: segStart + float64(j)*step, Reliable: j == 0})
		}
	}
	const off = 1.8
	subs := shift(cues, func(t float64) float64 { return t - off })
	r2 := analyze(subs, words)
	if r2.Status != "needs_fix" || len(r2.Mapping.Pieces) != 1 {
		t.Fatalf("got %s %s", r2.Status, r2.Message)
	}
	if got := r2.Mapping.Pieces[0].Offset; math.Abs(got-off) > 0.05 {
		t.Fatalf("offset %.3f, want %.3f (%s)", got, off, r2.Message)
	}
}
