#!/usr/bin/env python3
# Tim Marvel

import collect_data_no_ml
import learn_from_data

# play and collect some data without any intelligence
# collect_data_no_ml.play_the_game(2, False)
# collect_data_no_ml.play_the_game(2, True)
learn_from_data.train_the_net(0, 100)
learn_from_data.train_the_net(0, 200)
learn_from_data.train_the_net(0, 500)
