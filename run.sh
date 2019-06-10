echo "Running Scripts for model training ..."
# # FCNs with NewZealand
# python src/train.py -root NewZealand -net FCN32s -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# python src/train.py -root NewZealand -net FCN16s -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# python src/train.py -root NewZealand -net FCN8s -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# U-Net with NewZealand
python src/train.py -root NewZealand -net UNet -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# SegNet with NewZealand
python src/train.py -root NewZealand -net SegNet -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# ResUNet with NewZealand
python src/train.py -root NewZealand -net ResUNet -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# MCFCN with NewZealand
python src/train.py -root NewZealand -net MCFCN -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# BRNetv0 with NewZealand
python src/train.py -root NewZealand -net BRNetv0 -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
# FPN with NewZealand
python src/train.py -root NewZealand -net FPN -trigger iter -interval 50 -terminal 5000 -batch_size 24 &&
echo "end"
