namespace NavSim.Runtime
{
    // The ONLY reward surface of M7 (spec sec 4): the +1 outcome and the per-step time cost, routed
    // per arm. selfish = scorer only (the structural floor); shared = the identical scalar copied to
    // both agents (naive cooperation); poca = group reward (counterfactual credit). PerStep time cost
    // routes the same way so the finish-fast incentive is arm-identical. No shaping exists anywhere.
    public static class ArmRouting
    {
        public enum Arm { Selfish = 0, Shared = 1, Poca = 2 }

        public struct Split { public float scorer; public float partner; public float group; }

        public static Split Outcome(Arm arm) => arm switch
        {
            Arm.Selfish => new Split { scorer = 1f },
            Arm.Shared  => new Split { scorer = 1f, partner = 1f },
            _           => new Split { group = 1f },
        };

        public static Split PerStep(Arm arm, float timeCost) => arm switch
        {
            Arm.Poca => new Split { group = timeCost },
            _        => new Split { scorer = timeCost, partner = timeCost },
        };
    }
}
