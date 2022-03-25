import tensorflow as tf
import numpy as np
import ltn
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="sfc_results.csv")
    parser.add_argument('--epochs',type=int,default=1000)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
EPOCHS = args['epochs']
csv_path = args['csv_path']

# Language

embedding_size = 10

g = {l:ltn.Constant(np.random.uniform(low=0.0,high=1.0,size=embedding_size),trainable=True) for l in 'abcdefghijklmn'}

Smokes = ltn.Predicate.MLP([embedding_size],hidden_layer_sizes=(16,16))
Friends = ltn.Predicate.MLP([embedding_size,embedding_size],hidden_layer_sizes=(16,16))

# define data + wrapper connectives/quantifiers
friends = [('a','b'), ('b','a'), ('b','c'), ('c','b'), ('c','d'),
  ('d','c'), ('a','e'), ('e','a'), ('e','f'), ('f','e'),
  ('f','a'), ('a','f'),('g','h'),('h','g'),('h','i'),
  ('i','h'),('i','j'),('j','i'),('k','l'),('l','k'),('m','i'),
  ('g','a'),('a','g'),('i','m'),('m','n')]
smokes = ['a','e','f','g','i']

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6),semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())

# defining the theory
@tf.function
def axioms(p_exists):
    """
    NOTE: we update the embeddings at each step
        -> we should re-compute the variables.
    """
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    axioms = []
    # Friends: knowledge incomplete in that
    #     Friend(x,y) with x<y may be known
    #     but Friend(y,x) may not be known

    axioms.append(formula_aggregator(
            [Friends([g[x],g[y]]) for (x,y) in friends]))
    axioms.append(formula_aggregator(
            [Not(Friends([g[x],g[y]])) for x in g for y in g if (x,y) not in friends and x<y ]))
    # Smokes: knowledge incomplete
    axioms.append(formula_aggregator(
            [Smokes(g[x]) for x in smokes]))
    axioms.append(formula_aggregator(
            [Not(Smokes(g[x])) for x in g if x not in smokes]))

    # friendship is anti-reflexive
    axioms.append(Forall(p,Not(Friends([p,p])),p=5))

    # friendship is symmetric
    axioms.append(Forall((p,q),Implies(Friends([p,q]),Friends([q,p])),p=5))

    # everyone has a friend
    axioms.append(Forall(p,Exists(q,Friends([p,q]),p=p_exists)))

    # smoking propagates among friends
    axioms.append(Forall((p,q),Implies(And(Friends([p,q]),Smokes(p)),Smokes(q))))

    # computing sat_level
    sat_level = formula_aggregator(axioms).tensor
    return sat_level

# Initialize all layers and the static graph.
print("Initial sat level %.5f"%axioms(p_exists=tf.constant(6.)))

# # Training
# 
# Define the metrics

metrics_dict = {
    'train_sat': tf.keras.metrics.Mean(name='train_sat'),
    'test_phi1': tf.keras.metrics.Mean(name='test_phi1'),
    'test_phi2': tf.keras.metrics.Mean(name='test_phi2'),
    'test_phi3': tf.keras.metrics.Mean(name='test_phi3')
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
trainable_variables = \
        Smokes.trainable_variables \
        + Friends.trainable_variables \
        + ltn.as_tensors(list(g.values()))

@tf.function
def train_step(p_exists):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(p_exists)
        loss = 1.-sat
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    metrics_dict['train_sat'](sat)

# friendOf is symmetric
@tf.function
def sat_phi1():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    phi1 = Forall((p,q),Implies(Friends([p,q]),Friends([q,p])),p=5)
    return phi1.tensor

# A friend of at least two smokers is also a smoker 
@tf.function
def sat_phi2():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    r = ltn.Variable.from_constants("r",list(g.values()))
    phi2 = Forall((p,q,r),Implies(
        And(
            And(Smokes(p), Smokes(r)), # p and q smoke
            And(Friends([p,r]), Friends([q,r]))  # r is friend of both p and q
        ),
        Smokes(r)),p=5)
    return phi2.tensor

# Two smokers, friends of the same person, are also friends
@tf.function
def sat_phi3():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    r = ltn.Variable.from_constants("r",list(g.values()))
    # phi3 = Forall((p,q,r),Implies(
    #     And(
    #         And(Smokes(p), Smokes(r)), # p and q smoke
    #         And(Friends([p,r]), Friends([q,r]))  # r is friend of both p and q
    #     ),
    #     Friends([p,q])),p=5)
    phi3 = Forall((p,q),Friends([p,q]))
    return phi3.tensor

@tf.function
def test_step():
    # sat
    metrics_dict['test_phi1'](sat_phi1())
    metrics_dict['test_phi2'](sat_phi2())
    metrics_dict['test_phi3'](sat_phi3())

track_metrics=20
template = "Epoch {}"
for metrics_label in metrics_dict.keys():
    template += ", %s: {:.4f}" % metrics_label
if csv_path is not None:
    csv_file = open(csv_path,"w+")
    headers = ",".join(["Epoch"]+list(metrics_dict.keys()))
    csv_template = ",".join(["{}" for _ in range(len(metrics_dict)+1)])
    csv_file.write(headers+"\n")

for epoch in range(EPOCHS):
    for metrics in metrics_dict.values():
        metrics.reset_states()

    if 0 <= epoch < 200:
        p_exists = tf.constant(1.)
    else:
        p_exists = tf.constant(6.)

    train_step(p_exists=p_exists)
    test_step()

    metrics_results = [metrics.result() for metrics in metrics_dict.values()]
    if epoch%track_metrics == 0:
        print(template.format(epoch,*metrics_results))
    if csv_path is not None:
        csv_file.write(csv_template.format(epoch,*metrics_results)+"\n")
        csv_file.flush()
if csv_path is not None:
    csv_file.close()


# PRINT 

def plt_heatmap(df, vmin=None, vmax=None):
    plt.pcolor(df, vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(0.5,len(df.index),1),df.index)
    plt.xticks(np.arange(0.5,len(df.columns),1),df.columns)
    plt.colorbar()


smokes_frame_before = pd.DataFrame(
        np.array([(x in smokes) for x in g]),
        columns=["smokes"],
        index=list('abcdefghijklmn'))
friend_frame_before = pd.DataFrame(
        np.array([[((x,y) in friends) for x in g] for y in g]),
        index = list('abcdefghijklmn'),
        columns = list('abcdefghijklmn'))

p = ltn.Variable.from_constants("p",list(g.values()))
q = ltn.Variable.from_constants("q",list(g.values()))

pred_friends = Friends([p,q]).tensor
pred_smokes = Smokes(p).tensor

friend_frame = pd.DataFrame(
        pred_friends.numpy(),
        index=list('abcdefghijklmn'),
        columns=list('abcdefghijklmn'))

smokes_frame = pd.DataFrame(
        pred_smokes.numpy(),
        index=list('abcdefghijklmn'),
        columns=["smokes(x)"])

np.set_printoptions(suppress=True)

pd.options.display.max_rows=999
pd.options.display.max_columns=999
pd.set_option('display.width',1000)
pd.options.display.float_format = '{:,.2f}'.format

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("smokes(x)")
plt_heatmap(smokes_frame_before, vmin=0, vmax=1)
plt.subplot(132)
plt.title("friendOf(x,y)")
plt_heatmap(friend_frame_before, vmin=0, vmax=1)

plt.savefig('ex_smokes_before.png')
plt.show()

plt.figure(figsize=(12,3))

plt.subplot(131)
plt.title("friendOf(x,y)")
plt_heatmap(friend_frame, vmin=0, vmax=1)
plt.subplot(132)
plt.title("smokes(x)")
plt_heatmap(smokes_frame, vmin=0, vmax=1)
plt.savefig('ex_smokes_inferfacts.png')
plt.show()