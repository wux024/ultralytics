# Define an array containing all the sub-optical field sizes to test
$subOpticalFieldSizes = @(128,64,32)
$dataset = "twoperson"

# Loop through each sub-optical field size
foreach ($size in $subOpticalFieldSizes) {

    # Build and run the command with the current sub-optical field size
    python tools/pose_val.py --dataset $dataset `
        --imgsz 640 `
        --models n `
        --model-type spipose `
        --optical-field-sizes 128 `
        --sub-optical-field-size $size `
        --imgsz-hadamard 256 `
        --seed 20241026 `
        --split test
}