function prediction = vggf_dropout(rate, varargin)
%VGGF_DROPOUT Returns a VGG-f model with dropout

  prediction = models.VGG8(varargin{:});
  
  fc3 = prediction.find(@vl_nnconv, -1);
  fc3.inputs{1} = vl_nndropout(fc3.inputs{1}, 'rate', rate);
  
  fc2 = prediction.find(@vl_nnconv, -2);
  fc2.inputs{1} = vl_nndropout(fc2.inputs{1}, 'rate', rate);

end

