package store

import (
	"database/sql"
	"path/filepath"
	"testing"
)

func open(t *testing.T) *Store {
	s, err := Open(filepath.Join(t.TempDir(), "t.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func claimOrder(t *testing.T, s *Store) []string {
	var out []string
	for {
		f, err := s.ClaimNext()
		if err != nil {
			t.Fatal(err)
		}
		if f == nil {
			return out
		}
		out = append(out, f.Path)
	}
}

func TestPriorityOrder(t *testing.T) {
	s := open(t)
	for _, p := range []string{"/m/bulk1", "/m/bulk2", "/m/bulk3"} {
		if r, _ := s.Enqueue(p, "sig", "manual all", false, PriorityBulk); r != "queued" {
			t.Fatalf("enqueue %s: %s", p, r)
		}
	}
	s.Enqueue("/m/recheck", "sig", "recheck", true, PriorityUser)
	s.Enqueue("/m/bazarr", "sig", "hook:bazarr", false, PriorityNow)
	// a hook for a file already waiting in the bulk queue moves it up
	if r, _ := s.Enqueue("/m/bulk3", "sig", "watcher", false, PriorityNow); r != "already" {
		t.Fatalf("re-enqueue: %s", r)
	}
	// a lower priority request never moves something down
	s.Enqueue("/m/bazarr", "sig", "manual all", false, PriorityBulk)

	got := claimOrder(t, s)
	// bulk3 was bumped to "next"; it keeps its original queue time, so it has waited longest
	want := []string{"/m/bulk3", "/m/bazarr", "/m/recheck", "/m/bulk1", "/m/bulk2"}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order %v, want %v", got, want)
		}
	}
}

func TestRequeuedFileTakesNewPriority(t *testing.T) {
	s := open(t)
	s.Enqueue("/m/a", "sig", "manual", false, PriorityNow)
	f, _ := s.ClaimNext()
	s.Finish(f.ID, InSync, "ok", "sig", nil, nil)
	s.Enqueue("/m/b", "sig", "manual all", false, PriorityBulk)
	// re-check of a finished file at user priority goes ahead of bulk work
	if r, _ := s.Enqueue("/m/a", "sig", "recheck", true, PriorityUser); r != "queued" {
		t.Fatalf("got %s", r)
	}
	if got := claimOrder(t, s); got[0] != "/m/a" {
		t.Fatalf("order %v", got)
	}
}

// Databases created before priorities existed are upgraded on open and keep their queue.
func TestMigrateOldDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.db")
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE TABLE files(
		id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, status TEXT NOT NULL DEFAULT 'new', prev_status TEXT,
		signature TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '', message TEXT NOT NULL DEFAULT '',
		offset_ms INTEGER, details TEXT, attempts INTEGER NOT NULL DEFAULT 0,
		queued_at REAL, started_at REAL, finished_at REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
		INSERT INTO files(path, status, queued_at, created_at, updated_at) VALUES('/m/old', 'queued', 1, 1, 1);`); err != nil {
		t.Fatal(err)
	}
	db.Close()

	s, err := Open(path)
	if err != nil {
		t.Fatalf("open old db: %v", err)
	}
	defer s.Close()
	s.Enqueue("/m/new", "sig", "watcher", false, PriorityNow)
	if got := claimOrder(t, s); len(got) != 2 || got[0] != "/m/new" || got[1] != "/m/old" {
		t.Fatalf("order %v", got)
	}
	// opening again must not fail (migration is idempotent)
	s2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	s2.Close()
}

func TestFinishOnlyAppliesToProcessing(t *testing.T) {
	s := open(t)
	s.Enqueue("/m/a", "sig", "x", false, PriorityNow)
	f, _ := s.ClaimNext()
	s.Finish(f.ID, InSync, "first", "sig", nil, nil)
	// a stale/duplicate result must not overwrite the recorded one
	s.Finish(f.ID, Fixed, "late duplicate", "sig2", nil, nil)
	if g, _ := s.Get(f.ID); g.Status != InSync || g.Message != "first" {
		t.Fatalf("got %s %q", g.Status, g.Message)
	}
}

func TestGiveUpAfterRepeatedUnfinishedAttempts(t *testing.T) {
	s := open(t)
	s.Enqueue("/m/crashy", "sig", "x", false, PriorityNow)
	for i := 0; i < MaxAttempts; i++ {
		f, _ := s.ClaimNext()
		if f == nil {
			t.Fatalf("attempt %d: nothing to claim", i)
		}
		s.ResetProcessing() // simulated crash/restart mid-job
	}
	if f, _ := s.ClaimNext(); f != nil {
		t.Fatalf("claimed again after %d unfinished attempts", MaxAttempts)
	}
	if g, _ := s.Get(1); g.Status != Failed {
		t.Fatalf("status %s", g.Status)
	}
}

func TestRequeueKeepsPlaceAndEngineDownDoesNotCount(t *testing.T) {
	s := open(t)
	s.Enqueue("/m/a", "sig", "x", false, PriorityNow)
	for i := 0; i < MaxAttempts+3; i++ { // engine outages never exhaust attempts
		f, _ := s.ClaimNext()
		if f == nil {
			t.Fatalf("round %d: gave up on an engine outage", i)
		}
		s.Requeue(f.ID, "engine down", false)
	}
}

func TestMarkMissingLeavesActiveFilesAlone(t *testing.T) {
	s := open(t)
	s.Enqueue("/m/a", "sig", "x", false, PriorityNow)
	s.ClaimNext()
	s.MarkMissing("/m/a")
	if st := s.StatusOf("/m/a"); st != Processing {
		t.Fatalf("status %s", st)
	}
	// so it can't be queued (and claimed by a second worker) while still processing
	if r, _ := s.Enqueue("/m/a", "sig", "watcher", false, PriorityNow); r != "already" {
		t.Fatalf("enqueue while processing: %s", r)
	}
}
