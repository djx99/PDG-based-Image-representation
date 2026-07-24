CUDA_VISIBLE_DEVICES="0" python evaluate.py \
--config_file ./logs_paper/gqgan/cifar-test1/config.yaml \
--ckpt_path ./logs_paper/gqgan/cifar-test1/checkpoints/epoch=29-step=23460.ckpt \
--batch_size 128