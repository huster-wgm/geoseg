#!/bin/bash
echo "Evaluating patch performances";
for root in NZ32km2 Vaihingen; do
    for net in FCN32s FCN16s FCN8s UNet SegNet ResUNet MCFCN BRNet FPN; do
        # checkpoint => FCN32-3*1*24-NZ32km2_iter_5000.pth
        if [ $root == NZ32km2 ]; then
            python src/testPatch.py -root $root -checkpoints $net-3*1*24-$root\_iter_5000.pth;
        else
            python src/testPatch.py -root $root -checkpoints $net-3*6*24-$root\_iter_5000.pth;
        fi
    done
done

echo "Evaluating area performances";
for root in NZ32km2 Vaihingen; do
    for net in FCN32s FCN16s FCN8s UNet SegNet ResUNet MCFCN BRNet FPN; do
        # checkpoint => FCN32-3*1*24-NZ32km2_iter_5000.pth
        if [ $root == NZ32km2 ]; then
            python src/testArea.py -root $root -checkpoints $net-3*1*24-$root\_iter_5000.pth;
        else
            python src/testArea.py -root $root -checkpoints $net-3*6*24-$root\_iter_5000.pth;
        fi
    done
done
