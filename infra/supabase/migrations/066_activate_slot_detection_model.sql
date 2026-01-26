-- Activate Slot Detection model in od_roboflow_models
-- This model should be visible in workflow model dropdowns

UPDATE od_roboflow_models
SET is_active = true
WHERE id = '2c690b09-1179-418e-bd7a-bb7dc7cbb284';

-- If the model has a different ID, activate by name pattern
UPDATE od_roboflow_models
SET is_active = true
WHERE name ILIKE '%slot%' OR display_name ILIKE '%slot%';
