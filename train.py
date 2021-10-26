#!/usr/bin/env python3
# Tim Marvel

import collect_data_no_ml
import collect_data_with_ml
import learn_from_data

# play and collect some data without any intelligence
# learn_from_data.train_the_net(4, 500)
# learn_from_data.evaluate_the_net(4, 500)
# collect_data_with_ml.play_the_game(2, 4, 500, False)
# collect_data_with_ml.play_the_game(10, 4, 5005, True)

'''
Dr Malinovic
Bagging and Boosting
check how this is done
'''

'''
learn_from_data.train_the_net(4, 5005)
learn_from_data.evaluate_the_net(4, 5005)

collect_data_with_ml.play_the_game(100, 4, 5005, False)
collect_data_with_ml.play_the_game(10, 4, 5005, True)
learn_from_data.train_the_net(4, 5005)
learn_from_data.evaluate_the_net(4, 5005)


learn_from_data.train_the_net(4, 5005)
learn_from_data.evaluate_the_net(4, 5005)


collect_data_no_ml.play_the_game(200, False)
collect_data_no_ml.play_the_game(40, True)
learn_from_data.train_the_net(0, 100)
learn_from_data.evaluate_the_net(0, 100)
learn_from_data.train_the_net(0, 200)
learn_from_data.evaluate_the_net(0, 200)
learn_from_data.train_the_net(0, 500)
learn_from_data.evaluate_the_net(0, 500)
learn_from_data.train_the_net(0, 800)
learn_from_data.evaluate_the_net(0, 800)

collect_data_with_ml.play_the_game(100, 0, 800, False)
collect_data_with_ml.play_the_game(20, 0, 800, True)

learn_from_data.train_the_net(1, 100)
learn_from_data.evaluate_the_net(1, 100)
learn_from_data.train_the_net(1, 200)
learn_from_data.evaluate_the_net(1, 200)
learn_from_data.train_the_net(1, 500)
learn_from_data.evaluate_the_net(1, 500)
learn_from_data.train_the_net(1, 800)
learn_from_data.evaluate_the_net(1, 800)
learn_from_data.train_the_net(1, 1500)
learn_from_data.evaluate_the_net(1, 1500)

collect_data_with_ml.play_the_game(50, 1, 1500, False)
collect_data_with_ml.play_the_game(10, 1, 1500, True)
learn_from_data.train_the_net(2, 2500)
learn_from_data.evaluate_the_net(2, 2500)
learn_from_data.train_the_net(2, 3500)
learn_from_data.evaluate_the_net(2, 3500)

collect_data_with_ml.play_the_game(100, 2, 3500, False)
collect_data_with_ml.play_the_game(20, 2, 3500, True)
learn_from_data.train_the_net(3, 2500)
learn_from_data.evaluate_the_net(3, 2500)
learn_from_data.train_the_net(3, 3500)
learn_from_data.evaluate_the_net(3, 3500)
collect_data_no_ml.play_the_game(50, False)
collect_data_no_ml.play_the_game(10, True)
learn_from_data.train_the_net(3, 5000)
learn_from_data.evaluate_the_net(3, 5000)

collect_data_with_ml.play_the_game(100, 3, 5000, False)
collect_data_with_ml.play_the_game(20, 3, 5000, True)
learn_from_data.train_the_net(4, 2500)
learn_from_data.evaluate_the_net(4, 2500)
learn_from_data.train_the_net(4, 3500)
learn_from_data.evaluate_the_net(4, 3500)
collect_data_no_ml.play_the_game(50, False)
collect_data_no_ml.play_the_game(10, True)
learn_from_data.train_the_net(4, 5000)
learn_from_data.evaluate_the_net(4, 5000)
collect_data_no_ml.play_the_game(100, False)
collect_data_no_ml.play_the_game(20, True)
learn_from_data.train_the_net(4, 5003)
learn_from_data.evaluate_the_net(4, 5003)
learn_from_data.train_the_net(4, 5004)
learn_from_data.evaluate_the_net(4, 5004)


collect_data_no_ml.play_the_game(50, False)
collect_data_no_ml.play_the_game(10, True)
'''
#collect_data_with_ml.play_the_game(2, 5, 500, False)
# collect_data_no_ml.play_the_game(50, False)
# collect_data_no_ml.play_the_game(10, True)

#collect_data_with_ml.play_the_game(50, 4, 500, False)
#collect_data_with_ml.play_the_game(12, 4, 500, True)

'''
learn_from_data.train_the_net(5, 700)
print("A")
learn_from_data.train_the_net(5, 1000)
print("B")
'''
#learn_from_data.train_the_net(5, 800)
# collect_data_with_ml.play_the_game(2, 5, 800, False)
# learn_from_data.train_the_net(5, 1000)
collect_data_with_ml.play_the_game(3, 5, 1000, False, 0.1)