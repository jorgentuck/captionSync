package whisper

import (
	"encoding/json"
	"testing"
)

func parse(t *testing.T, js string) *Chunk {
	var v verboseJSON
	if err := json.Unmarshal([]byte(js), &v); err != nil {
		t.Fatal(err)
	}
	return v.chunk()
}

func TestChunkSegmentLevel(t *testing.T) {
	c := parse(t, `{"language":"en","segments":[
		{"start":1.0,"end":3.0,"text":" Hello there, friend.","words":[{"word":" Hel","start":9,"end":9.1},{"word":"lo","start":9.1,"end":9.2}]},
		{"start":4.0,"end":5.0,"text":" [BLANK_AUDIO]"},
		{"start":6.0,"end":8.0,"text":" (laughs) Okay, fine."},
		{"start":9.0,"end":10.0,"text":" - Right."}]}`)
	if !c.SegmentLevel {
		t.Fatal("should be segment level")
	}
	var got []string
	for _, w := range c.Words {
		got = append(got, w.Text)
		if w.Text == "BL" || w.Text == "ANK" || w.Text == "BLANK_AUDIO" {
			t.Fatalf("non-speech token kept: %q", w.Text)
		}
	}
	if len(got) != 6 || got[0] != "Hello" || got[3] != "Okay," {
		t.Fatalf("words %v", got)
	}
	if !c.Words[0].Reliable || c.Words[0].Start != 1.0 {
		t.Fatalf("first word should be the reliable segment start: %+v", c.Words[0])
	}
	if c.Words[3].Reliable {
		t.Fatal("a segment led by (laughs) must not give 'Okay' a reliable start")
	}
	if !c.Words[5].Reliable {
		t.Fatal("a dialogue dash should not cost the segment its anchor")
	}
}

func TestChunkWordLevel(t *testing.T) {
	c := parse(t, `{"words":[{"word":"Hi","start":1.2,"end":1.4},{"word":"[Music]","start":2,"end":3}]}`)
	if c.SegmentLevel || len(c.Words) != 1 || !c.Words[0].Reliable {
		t.Fatalf("got %+v", c)
	}
}
