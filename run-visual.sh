echo "Running Scripts for visualization ..."
# vis area visualization
python visArea.py -target segmap \
-checkpoints FCN8s_iter_5000.pth UNet_iter_5000.pth \
-cuda True &&
# vis corner visualization
python visCorner.py -target segmap \
-checkpoints FCN8s_iter_5000.pth UNet_iter_5000.pth \
-cuda True &&
# vis corner comparison visualization
python visCornerComparison.py -buffer 2 -sizes 6 8 -target segmap \
-checkpoints FCN8s_iter_5000.pth UNet_iter_5000.pth \
-cuda True &&
# vis single visualization
python visSingle.py \
-checkpoints FCN8s_iter_5000.pth FCN16s_iter_5000.pth FCN32s_iter_5000.pth \
UNet_iter_5000.pth FPN_iter_5000.pth SegNet_iter_5000.pth \
MC-FCN_iter_5000.pth BR-Net_iter_5000.pth \
-cuda True &&
# vis single comparison visualization
python visSingleComparison.py -eval_fn jaccard \
-target segmap \
-significance 0.04 \
-checkpoints FCN32s_iter_5000.pth FCN16s_iter_5000.pth FCN8s_iter_5000.pth \
-cuda True &&
python visSingleComparison.py -eval_fn jaccard \
-target edge \
-significance 0.04 \
-checkpoints FCN32s_iter_5000.pth FCN16s_iter_5000.pth FCN8s_iter_5000.pth \
-cuda True &&
## FCN8s, U-Net, MC-FCN, BR-Net
python visSingleComparison.py -eval_fn jaccard \
-target segmap \
-significance 0.02 \
-checkpoints FCN8s_iter_5000.pth SegNet_iter_5000.pth UNet_iter_5000.pth FPN_iter_5000.pth MC-FCN_iter_5000.pth BR-Net_iter_5000.pth \
-cuda True &&
python visSingleComparison.py -eval_fn jaccard \
-target edge \
-significance 0.02 \
-checkpoints FCN8s_iter_5000.pth SegNet_iter_5000.pth UNet_iter_5000.pth FPN_iter_5000.pth MC-FCN_iter_5000.pth BR-Net_iter_5000.pth \
-cuda True &&
echo "end"
