%%
figure;
cc=hsv(3);
for i = 1 : MaxClass
    plot(TrainSet(logical(TrainTarget == i), 1));
    hold on;
end