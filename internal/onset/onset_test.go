package onset

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// synthetic: background bed everywhere, "speech" bursts (voice-band tones) at known times
func synth(bg float64, bursts [][2]float64, dur float64) []byte {
	r := rand.New(rand.NewSource(1))
	n := int(dur * sampleRate)
	pcm := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		t := float64(i) / sampleRate
		x := bg * (r.Float64()*2 - 1)
		for _, b := range bursts {
			if t >= b[0] && t < b[1] {
				x += 0.3 * (math.Sin(2*math.Pi*700*t) + 0.5*math.Sin(2*math.Pi*1300*t)) * (0.6 + 0.4*math.Sin(2*math.Pi*4*t))
			}
		}
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(math.Max(-1, math.Min(1, x))*32000)))
	}
	return pcm
}

func TestRefineFindsOnsetAfterEarlyGuess(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{2.0, 4.5}, {7.3, 9.0}}, 12))
	// whisper put the second line's start at 4.6 (right after the first line ended)
	got, ok := d.Refine(4.6, 9.5)
	if !ok || math.Abs(got-7.3) > 0.05 {
		t.Fatalf("got %.3f ok=%v, want ~7.3", got, ok)
	}
}

func TestRefineSlightlyLateGuess(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{3.0, 5.0}}, 8))
	if got, ok := d.Refine(3.2, 6); !ok || math.Abs(got-3.0) > 0.05 {
		t.Fatalf("got %.3f ok=%v, want ~3.0", got, ok)
	}
}

func TestRefineNoContrast(t *testing.T) {
	d := New(synth(0.6, nil, 6)) // loud noise only, no speech-like onset
	if _, ok := d.Refine(2, 5); ok {
		t.Fatal("found an onset in pure noise")
	}
}

// review: a noisy gap between two lines must not let the back-up walk into the previous line
func TestRefineDoesNotWalkIntoPreviousLine(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{3.0, 4.0}, {4.6, 6.0}}, 9))
	got, ok := d.Refine(4.1, 7.0)
	if ok && math.Abs(got-4.6) > 0.06 {
		t.Fatalf("got %.3f, want ~4.6 or not ok", got)
	}
}

// review: whisper start inside running speech -> don't jump to the next phrase
func TestRefineInsideSpeechIsNotOK(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{1.0, 5.0}, {6.0, 8.0}}, 10))
	if got, ok := d.Refine(3.0, 9.0); ok {
		t.Fatalf("returned %.3f ok inside running speech", got)
	}
}

// review: a short transient (click/music hit) before the line is not the onset
func TestRefineIgnoresShortTransient(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{2.0, 2.06}, {3.5, 5.5}}, 8))
	if got, ok := d.Refine(1.0, 7.0); !ok || math.Abs(got-3.5) > 0.06 {
		t.Fatalf("got %.3f ok=%v, want ~3.5", got, ok)
	}
}

// review: line after a long silence (whisper's start at the previous line's end, > 4 s early)
func TestRefineAfterLongSilence(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{1.0, 2.0}, {9.0, 11.0}}, 14))
	if got, ok := d.Refine(2.1, 13.0); !ok || math.Abs(got-9.0) > 0.06 {
		t.Fatalf("got %.3f ok=%v, want ~9.0", got, ok)
	}
}

// round 2: whisper start inside the previous line's last ~100 ms still finds the next onset
func TestRefineStartInPreviousLineTail(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{2.0, 4.5}, {7.3, 9.5}}, 12))
	for _, st := range []float64{4.3, 4.4} {
		if got, ok := d.Refine(st, 11); !ok || math.Abs(got-7.3) > 0.06 {
			t.Fatalf("t=%.1f: got %.3f ok=%v, want ~7.3", st, got, ok)
		}
	}
}

// round 2: a 150-300 ms sound before the line is not taken as the onset
func TestRefineIgnoresMediumTransient(t *testing.T) {
	for _, burst := range []float64{0.15, 0.2, 0.25} {
		d := New(synth(0.02, [][2]float64{{2.0, 2.0 + burst}, {3.0, 5.5}}, 8))
		if got, ok := d.Refine(1.5, 6); ok && math.Abs(got-3.0) > 0.06 {
			t.Fatalf("burst %.2f: took %.3f", burst, got)
		}
	}
}

// round 2: fast dialogue - start inside a line, next phrase after a short pause -> not ok.
// (A start in the line's last moments is the "previous line's tail" case, where the next onset is right.)
func TestRefineFastDialogue(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{1.0, 3.0}, {3.4, 6.0}}, 8))
	for st := 1.1; st < 2.7; st += 0.05 {
		if got, ok := d.Refine(st, 7); ok && got > st+0.2 {
			t.Fatalf("t=%.2f jumped to later phrase %.3f", st, got)
		}
	}
}

// round 2: a lone sound effect in a long silence is not an onset far from whisper's start
func TestRefineLoneSFXFarAway(t *testing.T) {
	d := New(synth(0.02, [][2]float64{{1.0, 2.0}, {9.0, 9.3}, {14.0, 16.0}}, 18))
	if got, ok := d.Refine(2.0, 17); ok && math.Abs(got-14.0) > 0.06 {
		t.Fatalf("took %.3f (SFX at 9.0)", got)
	}
}
