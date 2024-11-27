# Define an array containing all the sub-optical field sizes to test
$subOpticalFieldSizes = @(64,32,16)
$dataset = "realworldperson"

# Loop through each sub-optical field size
foreach ($size in $subOpticalFieldSizes) {

    # Build and run the command with the current sub-optical field size
    python tools/pose_stream_inference.py --dataset $dataset `
        --imgsz 640 `
        --models n `
        --model-type spipose `
        --optical-field-sizes 128 `
        --kpt-line `
        --sub-optical-field-size $size `
        --inverse `
        --seed 20241122 `
}