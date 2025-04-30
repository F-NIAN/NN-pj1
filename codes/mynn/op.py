from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        X = self.input
        batch_size = X.shape[0]

        self.grads['W'] = np.matmul(X.T, grad) / batch_size  
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        if self.weight_decay:# 添加L2正则化项的梯度
            self.grads['W'] += self.weight_decay_lambda * self.W
        grad_prev = np.matmul(grad, self.W.T) # 计算返回前一层的梯度
        return grad_prev
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) #支持卷核的元组形式和正方形尺寸形式
        self.stride = stride
        self.padding = padding
        
        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))
        self.b = initialize_method(size=(out_channels))
        
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]    ->W[out,in,k,k] and b[out,]
        no padding
        """
        
        batch_size, in_channels, H, W = X.shape
        kH, kW = self.kernel_size
        
        new_H = (H + 2 * self.padding - kH) // self.stride + 1
        new_W = (W + 2 * self.padding - kW) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))
        
        for i in range(new_H):
            for j in range(new_W):
                h_start = i * self.stride
                h_end = h_start + kH
                w_start = j * self.stride
                w_end = w_start + kW
                window = X[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.tensordot(window, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b #(b,in,k,k)*(out,in,k,k)+(out,)
        
        self.input = X
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        batch_size, in_channels, H, W = X.shape
        out_channels, _, kH, kW = self.W.shape
        _, _, out_H, out_W = grads.shape
        
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)
        
        for b in range(batch_size):
            for c_out in range(out_channels):
                db[c_out] += np.sum(grads[b, c_out])
                for h in range(out_H):
                    for w in range(out_W):
                        h_start = h * self.stride
                        h_end = h_start + kH
                        w_start = w * self.stride
                        w_end = w_start + kW
                        dW[c_out] += X[b, :, h_start:h_end, w_start:w_end] * grads[b, c_out, h, w]
                        dX[b, :, h_start:h_end, w_start:w_end] += self.W[c_out] * grads[b, c_out, h, w]
        
        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size
        
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  
        self.grads = None
        self.predicts = None
        self.labels = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        batch_size = predicts.shape[0]
        correct_probs = probs[np.arange(batch_size), labels]
        
        correct_probs = np.clip(correct_probs, 1e-15, 1.0) #avoid log0
        loss = -np.mean(np.log(correct_probs))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]

        if self.has_softmax:# probs - one_hot_labels
            probs = softmax(self.predicts)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(batch_size), self.labels] = 1
            self.grads = (probs - one_hot) / batch_size
        else:
            grads = np.zeros_like(self.predicts)
            correct_probs = self.predicts[np.arange(batch_size), self.labels]
            correct_probs = np.clip(correct_probs, 1e-15, 1.0)
            grads[np.arange(batch_size), self.labels] = -1.0 / correct_probs
            self.grads = grads / batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, layer, reg_coeff=0.01):
        super().__init__()
        self.layer = layer        
        self.reg_coeff = reg_coeff  
        self.l2_loss = 0.0          
        self.params = {'W': layer.W, 'b': layer.b}
        self.weight_decay = layer.weight_decay
        self.weight_decay_lambda = layer.weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        output = self.layer(X)
        self.l2_loss = 0.0
        
        for param in self.layer.params.values():
            self.l2_loss += np.sum(param ** 2) 
            
        self.l2_loss = 0.5 * self.reg_coeff * self.l2_loss  
            
        return output

    def backward(self, grad):
        reg_grads = {}
        for name, param in self.layer.params.items():
            reg_grads[name] = self.reg_coeff * param  

        for name in self.layer.grads.keys():
            if self.layer.grads[name] is not None:
                self.layer.grads[name] += reg_grads[name]
            else:
                self.layer.grads[name] = reg_grads[name]
                
        return self.layer.backward(grad)
    
    def clear_grad(self):
        self.layer.clear_grad()
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition



class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.optimizable = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch, channels, H, W = X.shape
        k = self.kernel_size
        H_out = (H - k) // self.stride + 1
        W_out = (W - k) // self.stride + 1
        
        output = np.zeros((batch, channels, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k
                region = X[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(region, axis=(2,3))
        return output

    def backward(self, grad):
        X = self.input
        batch, channels, H, W = X.shape
        k = self.kernel_size
        H_out = grad.shape[2]
        W_out = grad.shape[3]
        
        dX = np.zeros_like(X)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k
                
                region = X[:, :, h_start:h_end, w_start:w_end]
                max_mask = (region == np.max(region, axis=(2,3), keepdims=True))
                dX[:, :, h_start:h_end, w_start:w_end] += max_mask * grad[:, :, i:i+1, j:j+1]
        return dX

class Flatten(Layer):
    def __init__(self):
        super().__init__() 
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)