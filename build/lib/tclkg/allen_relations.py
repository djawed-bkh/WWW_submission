from .allen_list import ALLEN_RELATIONS as arl


class AllenRelation:
    def __init__(self, quadruplet_A, quadruplet_B) -> None:
        """
        Initialise une relation Allen entre deux intervalles.
        quadruplet_A et quadruplet_B doivent avoir un attribut .date
        qui est une Interval avec .start et .end (ou tuple (start, end))
        """

        interval_A = quadruplet_A.date
        interval_B = quadruplet_B.date


        if hasattr(interval_A, "start") and hasattr(interval_A, "end"):
            self.A_start = interval_A.start
            self.A_end = interval_A.end
        else:
            self.A_start, self.A_end = interval_A

        if hasattr(interval_B, "start") and hasattr(interval_B, "end"):
            self.B_start = interval_B.start
            self.B_end = interval_B.end
        else:
            self.B_start, self.B_end = interval_B

    def check_one_axiom(self, axiom_name: str) -> bool:
        if axiom_name not in arl:
            raise ValueError(f"Axiom '{axiom_name}' is not recognized.")

        if axiom_name == "equals":
            return self.is_A_equal_B()
        elif axiom_name == "before":
            return self.is_A_before_B()
        elif axiom_name == "after":
            return self.is_A_after_B()
        elif axiom_name == "meets":
            return self.is_A_meets_B()
        elif axiom_name == "met_by":
            return self.is_A_met_by_B()
        elif axiom_name == "overlaps":
            return self.is_A_overlaps_B()
        elif axiom_name == "overlapped_by":
            return self.is_A_overlapped_by_B()
        elif axiom_name == "during":
            return self.is_A_during_B()
        elif axiom_name == "contains":
            return self.is_A_contains_B()
        elif axiom_name == "starts":
            return self.is_A_starts_B()
        elif axiom_name == "started_by":
            return self.is_A_started_by_B()
        elif axiom_name == "finishes":
            return self.is_A_finishes_B()
        elif axiom_name == "finished_by":
            return self.is_A_finished_by_B()
        else:
            raise ValueError(f"Axiom '{axiom_name}' is not implemented.")

    def check_all_axioms(self) -> dict:
        results = {}
        results["equals"] = self.is_A_equal_B()
        results["before"] = self.is_A_before_B()
        results["after"] = self.is_A_after_B()
        results["meets"] = self.is_A_meets_B()
        results["met_by"] = self.is_A_met_by_B()
        results["overlaps"] = self.is_A_overlaps_B()
        results["overlapped_by"] = self.is_A_overlapped_by_B()
        results["during"] = self.is_A_during_B()
        results["contains"] = self.is_A_contains_B()
        results["starts"] = self.is_A_starts_B()
        results["started_by"] = self.is_A_started_by_B()
        results["finishes"] = self.is_A_finishes_B()
        results["finished_by"] = self.is_A_finished_by_B()
        return results


    def is_A_equal_B(self) -> bool:
        return (self.A_start == self.B_start) and (self.A_end == self.B_end)


    def is_A_before_B(self) -> bool:
        return self.A_end < self.B_start


    def is_A_after_B(self) -> bool:
        return self.B_end < self.A_start


    def is_A_meets_B(self) -> bool:
        return self.A_end == self.B_start


    def is_A_met_by_B(self) -> bool:
        return self.B_end == self.A_start


    def is_A_overlaps_B(self) -> bool:
        return (
            (self.A_start < self.B_start)
            and (self.A_end > self.B_start)
            and (self.A_end < self.B_end)
        )


    def is_A_overlapped_by_B(self) -> bool:
        return (
            (self.B_start < self.A_start)
            and (self.B_end > self.A_start)
            and (self.B_end < self.A_end)
        )


    def is_A_during_B(self) -> bool:
        return (self.A_start > self.B_start) and (self.A_end < self.B_end)


    def is_A_contains_B(self) -> bool:
        return (self.B_start > self.A_start) and (self.B_end < self.A_end)


    def is_A_starts_B(self) -> bool:
        return (self.A_start == self.B_start) and (self.A_end < self.B_end)


    def is_A_started_by_B(self) -> bool:
        return (self.B_start == self.A_start) and (self.B_end < self.A_end)


    def is_A_finishes_B(self) -> bool:
        return (self.A_start > self.B_start) and (self.A_end == self.B_end)


    def is_A_finished_by_B(self) -> bool:
        return (self.B_start > self.A_start) and (self.B_end == self.A_end)
