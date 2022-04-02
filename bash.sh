
for style_img in style_images/Aligned/*
    do
        python train.py --style_img ${style_img##*/}
    done