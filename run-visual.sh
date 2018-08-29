echo "Running Scripts for visualization ..."
# gen area visualization
python genArea.py -target segmap \
-checkpoints FCN8s_iter_10000.pth UNet_iter_10000.pth \
-cuda True &&
# gen corner visualization
python genCorner.py -target segmap \
-checkpoints FCN8s_iter_10000.pth UNet_iter_10000.pth \
-cuda True &&
# gen corner comparison visualization
python genCornerComparison.py -buffer 2 -sizes 6 8 -target segmap \
-checkpoints FCN8s_iter_10000.pth UNet_iter_10000.pth \
-cuda True &&
# gen single visualization
python genSingle.py \
-checkpoints FCN8s_iter_10000.pth UNet_iter_10000.pth mtFCNv0-alpha-0.5_iter_10000.pth \
-cuda True &&
# gen single comparison visualization
python genSingleComparison.py -eval_fn jaccard \
-target segmap \
-significance 0.04 \
-checkpoints FCN8s_iter_10000.pth UNet_iter_10000.pth \
-cuda True &&
echo "end"
