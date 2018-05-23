function images = small_image_augmentation(images, shift)
%SMALL_IMAGE_AUGMENTATION data augmentation for small images (MNIST, CIFAR)

  n = size(images,4);
  flip = randi([0, 1], n, 1);
  shift_y = randi([-shift, shift], n, 1);
  shift_x = randi([-shift, shift], n, 1);

  for i = 1:n
    im = images(:,:,:,i);
    
    % flip it horizontally
    if flip(i)
      im = fliplr(im);
    end
    
    % circularly shift
    y = shift_y(i);
    x = shift_x(i);
    im = circshift(im, [y, x]);
    
    % fill with 0 the pixels that were cycled around the edge
    if y > 0
      im(1:y,:,:) = 0;
    else
      im(end+y+1:end,:,:) = 0;
    end
    if x > 0
      im(:,1:x,:) = 0;
    else
      im(:,end+x+1:end,:) = 0;
    end
    
    images(:,:,:,i) = im;
  end
end
