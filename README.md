# GraphBasedSparseKeypointTracking


# Before getting started make sure to follow the DinoTracker and PoseAnything README. (See related URL's at the buttom of this file)

Prepare pseudo labels:
# Use PoseAnything repository to create the psuedo labels.
    # python PoseAnything/psuedo_labels_generator.py \
  --support path/to/support.jpg \
  --queries path/to/query1.jpg path/to/query2.jpg \
  --config configs/pose_model_config.py \
  --checkpoint checkpoints/model.pth \
  --outdir output

The script psuedo_labels_generator.py generates the output to :
/home/adminubuntu/Lumenova/
├── PoseAnything/
│   └── readyoutputs/
│       └── <sequence_name>/
│           └── pose_results.json

# DinoTracker-GNN expects differnect coridnates and adj matrics. Run
    # python scripts/run_pose_pipeline.py <sequence_name>

The run_pose_pipeline script provides as follows:
# Adjusted JSON file with the keypoints on each frame and the adjacency matrix
# output video based on the psuedo labels
# Annotated frames

For DinoTracker-GNN you need to first train the original DinoTracker and copy the weights files to the apropriate location according to the DinoTracker README

Now we you are ready to train the GNN based DinoTracker.

Our code is based on code from:
https://github.com/orhir/PoseAnything
https://github.com/AssafSinger94/dino-tracker
