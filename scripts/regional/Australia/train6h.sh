export NCCL_ALGO=Tree
export OUTPUT_DIR=./Australia_6h
export CUDA_VISIBLE_DEVICES=0,1,2,3

python src/weatherode/regional_forecast/train.py --config configs/regional_forecast_weatherode.yaml \
    --trainer.strategy=ddp \
    --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --trainer.logger.name=predict_6h_Australia \
    --data.root_dir=/jupyter/weather_prediction/weather_prediction/5.625deg_npz \
    --data.predict_range=6 \
    --model.pretrained_path='' --data.out_variables=["geopotential_500","temperature_850",'2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind'] \
    --model.lr=5e-4 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5 \
    --model.ode_lr=1e-4 \
    --model.net.time_steps=6 \
    --model.net.patch_size=2 \
    --model.net.img_size=[10,14] \
    --model.net.depth=4 \
    --model.net.err_type=3D \
    --model.net.predict_list=[6] \
    --model.net.gradient_loss=False \
    --model.net.err_with_x=True \
    --model.net.err_with_v=True \
    --data.batch_size=8 \
    --data.region=Australia \
    --model.warmup_epochs=20000 \
    --model.max_epochs=80000 \
