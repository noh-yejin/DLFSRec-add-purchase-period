python main.py \
    --model_name="period_commerce" \
    --epochs=100 \
    --patience=40 \
    --fusion_type="concat"--hidden_size=64  \
    --data_dir='./data/e_commerce_cosmetic/' \
    --data_name='e_commerce_cosmetic' \
    --e_commerce=True \
    --period_add_like_cxt=True