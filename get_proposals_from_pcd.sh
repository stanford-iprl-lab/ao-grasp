#!/bin/bash
PCD_PATH="$1"

AO_CONDA_ENV="ao-grasp" # TODO: CHANGE TO YOUR AO-GRASP CONDA ENV NAME
CGN_CONDA_ENV="cgn" # TODO: CHANGE TO OUR CGN CONDA ENV NAME

CGN_ROOT_DIR="./contact_graspnet"

OUTPUT_DIR="./output"

# Get heatmap prediction
echo "Running run_pointscore_inference.py to get heatmap predictions..."
conda run -n $AO_CONDA_ENV python run_pointscore_inference.py --pcd_path $PCD_PATH --output_dir $OUTPUT_DIR

# Get heatmap path
filename=$(basename -- "$PCD_PATH")
extension="${filename##*.}"
data_name="${filename%.*}"
heatmap_path=$OUTPUT_DIR"/point_score/"$data_name".npz"

if [ ! -f $heatmap_path ]; then
    echo "Heatmap file" $heatmap_path "not found! Heatmap inference may have failed. Exiting..."
    exit 1
fi

echo "Getting grasp orientations..."

# Generate and save grasp proposals
conda run -n $CGN_CONDA_ENV python contact_graspnet/contact_graspnet/run_cgn_on_heatmap_file.py $heatmap_path --viz_top_k 10 --viz_save_as_mp4

proposal_path=$OUTPUT_DIR"/grasp_proposals/"$data_name".npz"
echo "Saved grasp proposals to "$proposal_path
