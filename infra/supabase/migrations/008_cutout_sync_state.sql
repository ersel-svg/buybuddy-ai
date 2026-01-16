-- Migration: Cutout Sync State
-- Description: Track min/max synced external IDs for incremental sync

-- Sync state table for tracking cutout synchronization progress
CREATE TABLE IF NOT EXISTS cutout_sync_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Synced external ID range
    min_synced_external_id INTEGER,  -- Oldest synced cutout ID
    max_synced_external_id INTEGER,  -- Newest synced cutout ID

    -- BuyBuddy API state (for reference)
    buybuddy_max_external_id INTEGER,  -- Max ID in BuyBuddy (last known)
    buybuddy_min_external_id INTEGER,  -- Min ID in BuyBuddy (last known)

    -- Progress tracking
    total_synced INTEGER DEFAULT 0,
    backfill_completed BOOLEAN DEFAULT false,

    -- Timestamps
    last_sync_new_at TIMESTAMPTZ,      -- Last "sync new" operation
    last_backfill_at TIMESTAMPTZ,      -- Last "backfill" operation
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert initial row (singleton pattern)
INSERT INTO cutout_sync_state (id)
VALUES ('00000000-0000-0000-0000-000000000001')
ON CONFLICT (id) DO NOTHING;

-- Function to update sync state after sync_new
CREATE OR REPLACE FUNCTION update_sync_state_new(
    p_max_external_id INTEGER,
    p_synced_count INTEGER
) RETURNS void AS $$
BEGIN
    UPDATE cutout_sync_state
    SET
        max_synced_external_id = GREATEST(COALESCE(max_synced_external_id, 0), p_max_external_id),
        min_synced_external_id = COALESCE(min_synced_external_id, p_max_external_id),
        total_synced = total_synced + p_synced_count,
        last_sync_new_at = NOW(),
        updated_at = NOW()
    WHERE id = '00000000-0000-0000-0000-000000000001';
END;
$$ LANGUAGE plpgsql;

-- Function to update sync state after backfill
CREATE OR REPLACE FUNCTION update_sync_state_backfill(
    p_min_external_id INTEGER,
    p_synced_count INTEGER,
    p_backfill_completed BOOLEAN DEFAULT false
) RETURNS void AS $$
BEGIN
    UPDATE cutout_sync_state
    SET
        min_synced_external_id = LEAST(COALESCE(min_synced_external_id, p_min_external_id), p_min_external_id),
        total_synced = total_synced + p_synced_count,
        backfill_completed = p_backfill_completed,
        last_backfill_at = NOW(),
        updated_at = NOW()
    WHERE id = '00000000-0000-0000-0000-000000000001';
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_cutout_sync_state_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER cutout_sync_state_updated_at
    BEFORE UPDATE ON cutout_sync_state
    FOR EACH ROW
    EXECUTE FUNCTION update_cutout_sync_state_timestamp();
