ps aux|grep train.py|grep -v grep|cut -c 9-16|xargs kill -9
ps aux|grep pretrain_sora.py|grep -v grep|cut -c 9-16|xargs kill -9
ps aux|grep get_wan_feature.py|grep -v grep|cut -c 9-16|xargs kill -9
