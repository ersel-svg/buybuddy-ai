-- Migration 025: Scan Requests System
-- Allows users to request product scans when products are not in the system

-- ============================================
-- SCAN REQUESTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS scan_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Product identification
    barcode VARCHAR(100) NOT NULL,
    product_name VARCHAR(500),
    brand_name VARCHAR(255),

    -- Reference images (stored as array of storage paths)
    reference_images TEXT[] DEFAULT '{}',

    -- Additional info
    notes TEXT,

    -- Requester info
    requester_name VARCHAR(255) NOT NULL,
    requester_email VARCHAR(255) NOT NULL,

    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),

    -- Completion tracking
    completed_at TIMESTAMPTZ,
    completed_by_product_id UUID REFERENCES products(id) ON DELETE SET NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_scan_requests_barcode ON scan_requests(barcode);
CREATE INDEX IF NOT EXISTS idx_scan_requests_status ON scan_requests(status);
CREATE INDEX IF NOT EXISTS idx_scan_requests_created_at ON scan_requests(created_at DESC);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_scan_requests_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER scan_requests_updated_at
    BEFORE UPDATE ON scan_requests
    FOR EACH ROW
    EXECUTE FUNCTION update_scan_requests_updated_at();

-- ============================================
-- RLS POLICY
-- ============================================
ALTER TABLE scan_requests ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON scan_requests FOR ALL USING (true);

-- ============================================
-- STORAGE BUCKET FOR REFERENCE IMAGES
-- ============================================
-- Note: Run this in Supabase dashboard or via API:
-- INSERT INTO storage.buckets (id, name, public) VALUES ('scan-request-images', 'scan-request-images', true);

-- ============================================
-- FUNCTION: Auto-complete scan requests when product is created
-- ============================================
CREATE OR REPLACE FUNCTION auto_complete_scan_request()
RETURNS TRIGGER AS $$
BEGIN
    -- When a new product is created or status changes to 'ready',
    -- check if there's a pending scan request for this barcode
    IF (TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND NEW.status = 'ready')) THEN
        UPDATE scan_requests
        SET
            status = 'completed',
            completed_at = NOW(),
            completed_by_product_id = NEW.id
        WHERE
            barcode = NEW.barcode
            AND status IN ('pending', 'in_progress');
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER product_auto_complete_scan_request
    AFTER INSERT OR UPDATE ON products
    FOR EACH ROW
    EXECUTE FUNCTION auto_complete_scan_request();

COMMENT ON TABLE scan_requests IS 'User requests for product scans when products are not in the system';
COMMENT ON FUNCTION auto_complete_scan_request IS 'Automatically marks scan requests as completed when matching product is created';
