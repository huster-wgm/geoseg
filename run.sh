#!/bin/bash
echo "Training basic models";
for root in NZ32km2 Vaihingen PotsdamRGB PotsdamIRRG; do
    for net in FCN32s FCN16s FCN8s UNet SegNet ResUNet MCFCN BRNet FPN; do
        python src/train.py -root $root -net $net -trigger iter -interval 50 -terminal 5000 -batch_size 24;
    done
done

echo "Training ensemble models";
NetArr = ("FCN8s UNet" "FCN8s FPN" "UNet FPN" "FCN8s UNet FPN");
for root in NZ32km2 Vaihingen PotsdamRGB PotsdamIRRG; do
    for net in ${NetArr[*]}; do
        python src/estrain.py -root $root -net $net -trigger iter -interval 50 -terminal 5000 -batch_size 24;
    done
done


echo "End.."