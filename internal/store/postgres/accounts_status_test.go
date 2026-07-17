package postgres

import (
	"testing"
	"time"
)

func TestDerivePoolStatus(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]any
		want string
	}{
		{"quota", map[string]any{"disabled_for_quota": true, "enabled": true}, "quota_disabled"},
		{"disabled", map[string]any{"enabled": false}, "disabled"},
		{"cooldown", map[string]any{"enabled": true, "in_cooldown": true}, "cooldown"},
		{"model_blocked", map[string]any{"enabled": true, "blocked_model_ids": []string{"grok-4.5"}}, "model_blocked"},
		{"expired", map[string]any{"enabled": true, "expired": true}, "expired"},
		{"normal", map[string]any{"enabled": true}, "normal"},
		{"raw_model_blocked", map[string]any{"enabled": true, "pool_status": "model_blocked", "blocked_model_ids": []string{}}, "model_blocked"},
	}
	for _, tc := range cases {
		got := derivePoolStatus(tc.in)
		if got != tc.want {
			t.Fatalf("%s: got %q want %q", tc.name, got, tc.want)
		}
	}
}

func TestActiveBlockedModelsExpires(t *testing.T) {
	now := time.Unix(1_700_000_100, 0)
	blocked := map[string]any{
		"alive": map[string]any{"until": float64(1_700_000_200)},
		"dead":  map[string]any{"until": float64(1_700_000_000)},
		"perm":  map[string]any{"reason": "nope"},
	}
	out := activeBlockedModels(blocked, now)
	if _, ok := out["dead"]; ok {
		t.Fatalf("expired block should drop: %#v", out)
	}
	if _, ok := out["alive"]; !ok {
		t.Fatalf("active block missing: %#v", out)
	}
	if _, ok := out["perm"]; !ok {
		t.Fatalf("permanent block missing: %#v", out)
	}
}
