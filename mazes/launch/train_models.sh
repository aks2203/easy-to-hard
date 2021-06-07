python train.py \
--model recur_resnet \
--width 2 \
--depth 28 \
--epochs 1 \
--train_batch_size 64 \
--test_batch_size 64 \
--model_path recur_28_1.pth \
--quick_test \
--test_mode max_conf \
--test_iterations 8

