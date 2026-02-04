from dataloader import AegisV1Loader, AegisV2Loader, BeaverTailsLoader, BingoGuardLoader, HarmBenchLoader

def test_instantiation_modes():
    print("🚀 开始多模式加载测试...\n")

    # aegis_v1_loader = AegisV1Loader("/data1/zez/OpenSafeGuard/datasets/aegis-v1")
    # aegis_v1_train_split = loader3.load(split="train")
    # aegis_v1_test_split = aegis_v1_loader.load(split="test")

    # aegis_v2_loader = AegisV2Loader("/data1/zez/OpenSafeGuard/datasets/aegis-v2")
    # aegis_v2_train_split = aegis_v2_loader.load(split="train")
    # aegis_v2_test_split = aegis_v2_loader.load(split="test")
    # aegis_v2_validation_split = aegis_v2_loader.load(split="validation")

    # beavertails_loader = BeaverTailsLoader("/data1/zez/OpenSafeGuard/datasets/beavertails")
    # beavertails_30k_train_split = beavertails_loader.load(split="30k_train")
    # beavertails_30k_test_split = beavertails_loader.load(split="30k_test")
    # beavertails_330k_train_split = beavertails_loader.load(split="330k_train")
    # beavertails_330k_test_split = beavertails_loader.load(split="330k_test")

    # bingoguard_loader = BingoGuardLoader("/data1/zez/OpenSafeGuard/datasets/bingoguard")
    # bingoguard_train_split = bingoguard_loader.load(split="train")

    harmbench_loader = HarmBenchLoader()
    import pdb; pdb.set_trace()


    # # ---------------------------------------------------------
    # # 场景 1: 完全默认 (使用注入的默认路径和默认 split)
    # # ---------------------------------------------------------
    # print("--- [场景 1: 完全默认] ---")
    # # 此时内部会自动选择 "nvidia/Aegis-AI-Content-Safety-Dataset-1.0" 和 "train"
    # loader1 = AegisV1Loader() 
    # data1 = loader1.load()
    # print(f"成功加载！路径: {loader1.path} | 样本数: {len(data1)}")
    # print(f"首条 ID: {data1[0].id}\n")
    # print(data1[0])


    # # ---------------------------------------------------------
    # # 场景 2: 自定义路径和 Split (直接传位置参数)
    # # ---------------------------------------------------------
    # print("--- [场景 2: 自定义路径和位置参数] ---")
    # # 模拟你需要的 data1/zez/safe/aegis-v1 路径
    # # 这里我们用原路径做演示，效果是一样的
    # loader2 = AegisV1Loader("/data1/zez/OpenSafeGuard/datasets/aegis-v1", "train")
    # data2 = loader2.load()
    # print(f"成功加载！路径: {loader2.path} | 样本数: {len(data2)}")
    # print(f"首条 ID: {data2[0].id}\n")
    # print(data2[0])


    # # ---------------------------------------------------------
    # # 场景 3: 混合使用 (初始化指定路径，load 动态覆盖 split)
    # # ---------------------------------------------------------
    # print("--- [场景 3: 混合使用 + 动态覆盖] ---")
    # loader3 = AegisV1Loader("/data1/zez/OpenSafeGuard/datasets/aegis-v1")
    # # 动态覆盖为 test split
    # data3 = loader3.load(split="test")
    # print(f"成功加载！路径: {loader3.path} | 目标 Split: test | 样本数: {len(data3)}")
    # print(f"首条 ID: {data3[0].id}")
    # print(data3[0])

    



if __name__ == "__main__":
    test_instantiation_modes()
