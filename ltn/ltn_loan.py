import math
import tensorflow as tf
import numpy as np
import ltn
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path', type=str, default="sfc_results.csv")
    parser.add_argument('--epochs', type=int, default=1500)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


args = parse_args()
EPOCHS = args['epochs']
csv_path = args['csv_path']

# Language

embedding_size = 10

g = {l: ltn.Constant(np.random.uniform(low=0.0, high=1.0, size=embedding_size),
                     trainable=True) for l in 'abcdefghijklmnopqrstuv'}

Educated = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(24, 24))
Wealthy = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(24, 24))
Married = ltn.Predicate.MLP(
    [embedding_size, embedding_size], hidden_layer_sizes=(24, 24))
Loan = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(24, 24))

# define data + wrapper connectives/quantifiers
educated = ['a', 'f', 'h', 's']
wealthy = ['a', 'c', 'd', 'g', 'o', 'p', 'v', 'u']
married = [('a', 'd'), ('d', 'a'), ('f', 'n'), ('n', 'f'), ('c', 'g'), ('g', 'c'), 
    # ('k', 'l'),
    ('l', 'k'), ('j', 'm'), ('m', 'j'), ('b','i'), ('i', 'b'), ('e', 'h'), ('h', 'e'),
    # ('o', 'p'), 
    ('p', 'o'), ('q', 'r'), ('r', 'q'), ('t', 's'), ('s', 't'), ('u', 'v'), ('v', 'u')]
loan = ['a', 'c', 'd', 'f', 
    # 'g',
    'h', 'o', 
    # 'p', 
    's', 'v',
    # 'u'
    ]

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(
    ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
Exists = ltn.Wrapper_Quantifier(
    ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(
    ltn.fuzzy_ops.Aggreg_pMeanError())

equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])

# defining the theory


@tf.function
def axioms(p_exists):
    """
    NOTE: we update the embeddings at each step
        -> we should re-compute the variables.
    """
    p = ltn.Variable.from_constants("p", list(g.values()))
    q = ltn.Variable.from_constants("q", list(g.values()))
    r = ltn.Variable.from_constants("r", list(g.values()))

    axioms = []

    # EDUCATED: ck
    axioms.append(formula_aggregator(
        [Educated(g[x]) for x in educated]))
    axioms.append(formula_aggregator(
        [Not(Educated(g[x])) for x in g if x not in educated]))

    # WEALTHY: ck
    axioms.append(formula_aggregator(
        [Wealthy(g[x]) for x in wealthy]))
    axioms.append(formula_aggregator(
        [Not(Wealthy(g[x])) for x in g if x not in wealthy]))

    # MARRIED: ck
    axioms.append(formula_aggregator(
        [Married([g[x], g[y]]) for (x, y) in married]))
    axioms.append(formula_aggregator(
        [Not(Married([g[x], g[y]])) for x in g for y in g if (x, y) not in married and x < y]))

    # LOANS: ik
    axioms.append(formula_aggregator(
        [Loan(g[x]) for (x) in loan]))

    # 1) marriage is symmetric
    axioms.append(Forall((p, q), Implies(
        Married([p, q]), Married([q, p])), p=5))
    # 2) marriage is anti-reflective
    axioms.append(Forall(p, Not(Married([p, p])), p=5))
    # 4) only married to one person
    axioms.append(Forall((p,q),Implies(Married([p, q]),Not(Exists(r,Married([p,r])))),p=5))
    # 5) Everyone is married 
    axioms.append(Forall(p, Exists(q, Married([p, q])), p=p_exists))
    # 6) wealth or education results in (not) getting a loan
    axioms.append(Forall(p, Implies(Or(Wealthy(p), Educated(p)), Loan(p))))
    axioms.append(
        Forall(p, Implies(And(Not(Wealthy(p)), Not(Educated(p))), Not(Loan(p))), p=5))

    # 5) Married people are both wealthy
    # axioms.append(Forall((p, q), Implies(
    #     And(Married([p, q]), Wealthy(p)), Wealthy(q))))
    
    # computing sat_level
    sat_level = formula_aggregator(axioms).tensor
    return sat_level


# Initialize all layers and the static graph.
# print("Initial sat level %.5f" % axioms(p_exists=tf.constant(6.)))

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
    Educated.trainable_variables \
    + Wealthy.trainable_variables \
    + Married.trainable_variables \
    + Loan.trainable_variables \
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

# When you're married to someone who is wealthy, chances on a loan are higher
@tf.function
def sat_phi1():
    p = ltn.Variable.from_constants("p", list(g.values()))
    q = ltn.Variable.from_constants("q", list(g.values()))
    phi1 = Forall((p, q), Implies(And(Married([p, q]),Wealthy(q)), Loan(p)))
    return phi1.tensor

# A friend of at least two smokers is also a smoker
@tf.function
def sat_phi2():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    phi2 = Forall((p, q), Implies(And(Married([p, q]),Educated(q)), Not(Loan(p))))
    return phi2.tensor

@tf.function
def sat_phi3():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    phi3 = Forall((p, q), Implies(And(Married([p, q]), Wealthy(p)), Wealthy(q)))
    return phi3.tensor

@tf.function
def test_step():
    # sat
    metrics_dict['test_phi1'](sat_phi1())
    metrics_dict['test_phi2'](sat_phi2())
    metrics_dict['test_phi3'](sat_phi3())


# track_metrics = 20
# template = "Epoch {}"
# for metrics_label in metrics_dict.keys():
#     template += ", %s: {:.4f}" % metrics_label
# if csv_path is not None:
#     csv_file = open(csv_path, "w+")
#     headers = ",".join(["Epoch"]+list(metrics_dict.keys()))
#     csv_template = ",".join(["{}" for _ in range(len(metrics_dict)+1)])
#     csv_file.write(headers+"\n")

# for epoch in range(EPOCHS):
#     for metrics in metrics_dict.values():
#         metrics.reset_states()

#     if 0 <= epoch < 200:
#         p_exists = tf.constant(1.)
#     else:
#         p_exists = tf.constant(6.)

#     train_step(p_exists=p_exists)
#     test_step()

#     metrics_results = [metrics.result() for metrics in metrics_dict.values()]
#     if epoch % track_metrics == 0:
#         print(template.format(epoch, *metrics_results))
#     if csv_path is not None:
#         csv_file.write(csv_template.format(epoch, *metrics_results)+"\n")
#         csv_file.flush()
# if csv_path is not None:
#     csv_file.close()


# PRINT

def plt_heatmap(df, vmin=None, vmax=None):
    plt.pcolor(df, vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar()


loan_frame_before = pd.DataFrame(
    np.array([[(x in educated), (x in wealthy), (x in loan) if x in g else math.nan] for x in g]),
    columns=["educated(x)","wealthy(x)", "loan(x)"],
    index=list('abcdefghijklmnopqrstuv'))
married_frame_before = pd.DataFrame(
    np.array([[((x, y) in married) for x in g] for y in g]),
    index=list('abcdefghijklmnopqrstuv'),
    columns=list('abcdefghijklmnopqrstuv'))

# p = ltn.Variable.from_constants("p", list(g.values()))
# q = ltn.Variable.from_constants("q", list(g.values()))

# pred_married = Married([p, q]).tensor
# pred_loan = Loan(p).tensor

# married_frame = pd.DataFrame(
#     pred_married.numpy(),
#     index=list('abcdefghijklmnopqrstuv'),
#     columns=list('abcdefghijklmnopqrstuv'))

# loan_frame = pd.DataFrame(
#     tf.stack([Educated(p).tensor,Wealthy(p).tensor, Loan(p).tensor],axis=1).numpy(),
#     columns=["educated(x)","wealthy(x)", "loan(x)"],
#     index=list('abcdefghijklmnopqrstuv'))
    

np.set_printoptions(suppress=True)

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.title("married(x,y)")
plt_heatmap(married_frame_before, vmin=0, vmax=1)
plt.subplot(122)
plt_heatmap(loan_frame_before, vmin=0, vmax=1)

plt.savefig('graphs_out/ltn_loan_facts.png')
plt.show()

# plt.figure(figsize=(8, 3))

# plt.subplot(121)
# plt.title("married(x,y)")
# plt_heatmap(married_frame, vmin=0, vmax=1)
# plt.subplot(122)
# plt_heatmap(loan_frame, vmin=0, vmax=1)
# plt.savefig('graphs_out/ltn_loan_results.png')
# plt.show()
