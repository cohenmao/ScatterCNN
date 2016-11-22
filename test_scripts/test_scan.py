import theano
import theano.tensor as T


def greater_than_element(X, y):

    res, _ = theano.scan(fn=lambda X, y: T.maximum(X, y),
                         sequences = X,
                         non_sequences = y,
                         n_steps=X.shape[0]
                         )
    return res

X = T.vector("X")
Y = T.vector("Y")
y = T.scalar("y")

res, _ = theano.scan(fn=lambda Y, X: T.maximum(X, Y),
                     sequences=Y,
                     non_sequences=X,
                     n_steps=Y.shape[0]
                     )

greater_than_vector = theano.function(inputs=[X, Y], outputs=T.flatten(res))

print(greater_than_vector([1, 2, 3, 4], [0, 2.5, 5]))