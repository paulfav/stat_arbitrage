start = "2019-01-01"
end = "2023-01-01" 



import multiprocessing

num_cores = multiprocessing.cpu_count() -1


risk_per_month = 0.85   
risk_per_pair = 0.01

enter_z = 2
exit_z = 1
stop_loss_z = 1

p_val_threshold = 0.0001