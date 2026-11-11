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
