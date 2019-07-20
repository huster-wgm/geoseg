#!/bin/bash
echo "Training basic models";
for root in NZ32km2 Vaihingen; do
    for net in FCN32s FCN16s FCN8s UNet SegNet ResUNet MCFCN BRNet FPN; do
        python src/train.py -root $root -net $net -trigger iter -interval 50 -terminal 5000 -batch_size 24;
    done
done

echo "End.."