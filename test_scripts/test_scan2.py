import theano
import theano.tensor as T
import numpy


input = T.ftensor4("input")
scale = T.fvector("scale")
orientation = T.fvector("orientation")
n = T.scalar("n")



def create_mutual_one_v_all(all_responses, single_response):
    mutual_responses, _ = theano.scan(fn=lambda X, Y: create_mutual_one_v_one(X, Y),
                                      sequences=all_responses,
                                      non_sequences=single_response,
                                      n_steps=all_responses.shape[0]
                                      )

    return mutual_responses


def create_mutual_one_v_one(resX, resY):
    return 0.5 * (resX + resY)


def is_greater(X, y):

    g, _ = theano.scan(fn=lambda X, y: X > y,
                       sequences=X,
                       non_sequences=y,
                       n_steps=X.shape[0])

    return g

def equal_elementwise(X, y):

    g, _ = theano.scan(fn=lambda X, y: abs(X-y)<0.5,
                       sequences=X,
                       non_sequences=y,
                       n_steps=X.shape[0]
                       )
    return g

def orthogonal_elmentwise(X, y, n):

    g, _ = theano.scan(fn=lambda X, y, n: abs(abs(X - y)-n*0.5) < 0.5,
                       sequences=X,
                       non_sequences=[y, n],
                       n_steps=X.shape[0]
                       )
    return g

mr, _ = theano.scan(fn=lambda Y, X: create_mutual_one_v_all(X, Y),
                    sequences=input,
                    non_sequences=input,
                    n_steps=input.shape[0]
                    )

same_scale, _ = theano.scan(fn=lambda Y, X: equal_elementwise(X, Y),
                            sequences=scale,
                            non_sequences=scale,
                            n_steps=scale.shape[0]
                            )

orthogonal, _ = theano.scan(fn=lambda Y, X, n: orthogonal_elmentwise(X, Y, n),
                            sequences=orientation,
                            non_sequences=[orientation, n],
                            n_steps=orientation.shape[0]
                            )

reshaped_mutual_response = T.reshape(mr, (mr.shape[0]*mr.shape[1], mr.shape[2], mr.shape[3], mr.shape[4]))


idx = T.arange(scale.shape[0])
upper_tri, _ = theano.scan(fn=lambda Y, X: is_greater(X, Y),
                           sequences=idx,
                           non_sequences=idx,
                           n_steps=idx.shape[0]
                           )
combined_cond = T.flatten(upper_tri * same_scale * orthogonal)
flattened = T.flatten(upper_tri)

filtered_mutual_responses = reshaped_mutual_response[T.nonzero(flattened)]

f = theano.function(inputs=[input], outputs=mr, allow_input_downcast=True)
print(numpy.shape(f(numpy.arange(8).reshape((4, 1, 1, 2)))))
#ff = theano.function(inputs=[scale, orientation, n], outputs=[upper_tri, same_scale, orthogonal, combined_cond], allow_input_downcast=True)
#res = ff(numpy.repeat(range(3,8), 6), numpy.tile(range(6), 5), 6)
#print(len([ii for ii in res[3] if ii==True]))

