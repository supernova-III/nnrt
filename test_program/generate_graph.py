import tensorflow as tf
import onnx
import tf2onnx

tf.config.optimizer.set_experimental_options({
    'disable_meta_optimizer': True,   # отключить Grappler
    'constant_folding': False,
    'layout_optimizer': False,
    'arithmetic_optimization': False,
    'remapping': False
})

def export_to_onnx(cf, file_name):
    graph_def = cf.graph.as_graph_def()

    onnx_model, _ = tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=[*(x.name for x in cf.inputs)],
        output_names=[*(x.name for x in cf.outputs)],
        opset=17
    )
    onnx.save(onnx_model, file_name)
    onnx.save(onnx_model, f"/mnt/c/dev/{file_name}")

@tf.function
def custom_graph(a, b):
    c = tf.multiply(a, b)
    d = tf.sin(c)
    e = tf.cos(c)
    e = tf.add(d, e)
    return e


cf = custom_graph.get_concrete_function(
    tf.TensorSpec([None], tf.float32, name="a"),
    tf.TensorSpec([None], tf.float32, name="b")
)


export_to_onnx(cf, "my_graph.onnx")
