export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
python distilled_trainer.py --checkpoint_dir ../models/Wan2.1-T2V-1.3B --t5_cpu

