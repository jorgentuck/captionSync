package pipeline

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/jorgentuck/captionSync/internal/config"
	"github.com/jorgentuck/captionSync/internal/store"
)

func newSvc(t *testing.T) (*Service, *store.Store) {
	st, err := store.Open(filepath.Join(t.TempDir(), "t.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	return NewService(&config.Config{Settle: time.Second}, st, nil, nil), st
}

func claimAs(t *testing.T, st *store.Store, want string) *store.File {
	for {
		f, err := st.ClaimNext()
		if err != nil || f == nil {
			t.Fatalf("expected to claim %s, got %v %v", want, f, err)
		}
		if f.Path == want {
			return f
		}
		st.Finish(f.ID, store.InSync, "", "sig", nil, nil)
	}
}

func attempts(st *store.Store, id int64) int { f, _ := st.Get(id); return f.Attempts }

// An outage after a success elsewhere must not burn every queued file's attempts.
func TestOutageDoesNotCountAttempts(t *testing.T) {
	s, st := newSvc(t)
	s.lastOK.Store(time.Now().UnixNano()) // some other file worked a moment ago
	s.lastOKID.Store(999)
	st.Enqueue("/m/a", "sig", "x", false, store.PriorityNow)
	for i := 0; i < 8; i++ { // engine keeps failing; nothing else succeeds in between
		f := claimAs(t, st, "/m/a")
		s.requeue(f, Outcome{Requeue: true, EngineDown: true, Message: "down"})
		s.engineDown.Store(0)
	}
	if f, _ := st.Get(1); f.Status != store.Queued || f.Attempts != 0 {
		t.Fatalf("outage burned attempts: %s attempts=%d", f.Status, f.Attempts)
	}
}

// A file whose audio breaks the engine while other files keep working is eventually given up.
func TestPoisonFileIsCounted(t *testing.T) {
	s, st := newSvc(t)
	st.Enqueue("/m/poison", "sig", "x", false, store.PriorityNow)
	f := claimAs(t, st, "/m/poison")
	s.requeue(f, Outcome{Requeue: true, EngineDown: true, Message: "crash"}) // first failure: not counted
	if attempts(st, f.ID) != 0 {
		t.Fatalf("first failure counted")
	}
	for i := 0; i < 3; i++ {
		time.Sleep(2 * time.Millisecond)
		s.lastOK.Store(time.Now().UnixNano()) // another file succeeded since
		s.lastOKID.Store(999)
		f = claimAs(t, st, "/m/poison")
		s.requeue(f, Outcome{Requeue: true, EngineDown: true, Message: "crash"})
	}
	if attempts(st, f.ID) != 3 {
		t.Fatalf("attempts %d, want 3 counted", attempts(st, f.ID))
	}
}
