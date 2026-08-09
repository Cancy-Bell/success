#!/usr/bin/sh

python3 src/run_AURC_token.py \
    --card_number=1 \
    --train \
    --crf \
    --epochs=30 \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --test_batch_size=32 \
    --early_stop_patience=5 \
    --save_predictions \
    --target_domain='In-Domain' \
#    --target_domain='Cross-Domain' \
