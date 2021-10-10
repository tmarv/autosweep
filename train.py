#!/usr/bin/env python3
# Tim Marvel

import collect_data_no_ml
import collect_data_with_ml
import learn_from_data

# play and collect some data without any intelligence
collect_data_no_ml.play_the_game(2, False)
collect_data_no_ml.play_the_game(2, True)
learn_from_data.train_the_net(0, 100)
learn_from_data.evaluate_the_net(0, 100)
learn_from_data.train_the_net(0, 200)
learn_from_data.evaluate_the_net(0, 200)
learn_from_data.train_the_net(0, 500)
learn_from_data.evaluate_the_net(0, 500)
learn_from_data.train_the_net(0, 800)
learn_from_data.evaluate_the_net(0, 800)
collect_data_with_ml.play_the_game(20, 0, 500, False)
collect_data_with_ml.play_the_game(10, 0, 500, True)
learn_from_data.train_the_net(1, 100)
learn_from_data.evaluate_the_net(1, 100)
learn_from_data.train_the_net(1, 200)
learn_from_data.evaluate_the_net(1, 200)
learn_from_data.train_the_net(1, 500)
learn_from_data.evaluate_the_net(1, 500)
learn_from_data.train_the_net(1, 800)
learn_from_data.evaluate_the_net(1, 800)