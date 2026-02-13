CONFIG=$1

# if [ -z "${MASTER_ADDR}" ] || [ -z "${MASTER_PORT}" ]; then
#     echo "错误：请设置 MASTER_ADDR 和 MASTER_PORT 环境变量"
#     exit 1
# fi

# TARGET="${MASTER_ADDR}:${MASTER_PORT}"
# echo "等待 ${TARGET} 可访问..."

# while ! nc -z "${MASTER_ADDR}" "${MASTER_PORT}"; do
#     echo "仍无法连接到 ${TARGET}，1秒后重试..."
#     sleep 1
# done

# echo "${TARGET} 已可访问，开始执行 torchrun..."


torchrun --nproc_per_node 8 \
         --nnodes 1 \
         scripts/train.py \
         $CONFIG \
         --launcher pytorch ${@:2}