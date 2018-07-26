
import dynet as dy

# TODO: make sure this contains only expressions from current computation graph version
diagnose_expressions = []

class DiagnoseContext(object):
  def __init__(self, component):
    self.component = component
  def __enter__(self):
    assert self.expr is None
    self.start_expr_count = dy.cg()  # TODO
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end_expr_count = dy.cg()  # TODO
    # TODO: add all new expressions

def diagnose(component, expr=None):
  if expr:
    diagnose_expressions.append((component, expr))
  else:
    return DiagnoseContext(component)


