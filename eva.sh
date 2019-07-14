#!/bin/bash
echo "Evaluating basic models";
for root in Vaihingen PotsdamRGB PotsdamIRRG; do
    for net in FCN32s FCN16s FCN8s UNet SegNet ResUNet MCFCN BRNet FPN; do
        # checkpoint => FCN32-3*1*24-NZ32km2_iter_5000.pth
        if [ $root == "NZ32km2" ]; then
            python src/test.py -root $root -checkpoints $net-3*1*24-$root\_iter_5000.pth;
        else
            python src/test.py -root $root -checkpoints $net-3*6*24-$root\_iter_5000.pth;
        fi
    done
done

# echo "Evaluating basic models";
# for root in NZ32km2 Vaihingen PotsdamRGB PotsdamIRRG; do
#     for net in FCN32s FCN16s FCN8s UNet SegNet ResUNet MCFCN BRNet FPN; do
#         python src/estrain.py -root $root -net $net -trigger iter -interval 50 -terminal 5000 -batch_size 24;
#     done
# done


echo "End.."