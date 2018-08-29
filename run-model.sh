echo "Running Scripts for model training ..."
## echo FCN8s
python FCNs.py -ver FCN32s -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
python FCNs.py -ver FCN16s -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
python FCNs.py -ver FCN8s -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
## echo U-Net
python UNet.py -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
## echo FPN
python FPN.py -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
## echo SegNet
python SegNet.py -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
## echo ResUNet
python ResUNet.py -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
## echo MC-FCN
python MC-FCN.py -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
## echo BR-Net
python BR-Net.py -trigger iter -interval 10 -terminal 1000 -batch_size 24 &&
echo "end"
