python main.py \
    --model_name="period_grocery_random" \
    --epochs=100 \
    --patience=40 \
    --fusion_type="concat"\
    --hidden_size=64  \
    --period_add_like_cxt=True \
    --grocery=True