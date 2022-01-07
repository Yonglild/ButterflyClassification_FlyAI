# Cutmix

# for i, (input, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)

#         input = input.cuda()
#         target = target.cuda()

#         r = np.random.rand(1)
#         if args.beta > 0 and r < args.cutmix_prob:
#             # generate mixed sample
#             lam = np.random.beta(args.beta, args.beta)
#             rand_index = torch.randperm(input.size()[0]).cuda()
#             target_a = target
#             target_b = target[rand_index]
#             bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
#             input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
#             # adjust lambda to exactly match pixel ratio
#             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
#             # compute output
#             output = model(input)
#             loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
#         else:
#             # compute output
#             output = model(input)
#             loss = criterion(output, target)

#         # measure accuracy and record loss
#         err1, err5 = accuracy(output.data, target, topk=(1, 5))

#         losses.update(loss.item(), input.size(0))
#         top1.update(err1.item(), input.size(0))
#         top5.update(err5.item(), input.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0 and args.verbose == True:
#             print('Epoch: [{0}/{1}][{2}/{3}]\t'
#                   'LR: {LR:.6f}\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
#                   'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
#                 epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
#                 data_time=data_time, loss=losses, top1=top1, top5=top5))

#     print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
#         epoch, args.epochs, top1=top1, top5=top5, loss=losses))

#     return losses.avg


# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2




# Mixup
# def mixup_data(x, y, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     reg_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()

#         inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
#                                                        args.alpha, use_cuda)
#         inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)

#         outputs = net(inputs)
#         loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
#         train_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
#                     + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         progress_bar(batch_idx, len(trainloader),
#                      'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
#                         100.*correct/total, correct, total))
#     return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)