from __future__ import annotations

import copy
from typing import Any

import numpy as np


class Interval:
    start: Any
    end: Any

    def __init__(self, start: Any, end: Any) -> None:
        if start is None and end is not None:
            start = end
        elif end is None and start is not None:
            end = start
        if start is None or end is None or start <= end:
            self.start = start
            self.end = end
        else:
            self.start = end
            self.end = start

    def __str__(self) -> str:
        return str((self.start, self.end))

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def _is_valid_operand(self, other: object) -> bool:
        return hasattr(other, "start") and hasattr(other, "end")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self.start == other.start) and (self.end == other.end)

    def get_start(self) -> Any:
        return self.start

    def get_end(self) -> Any:
        return self.end

    def update_start(self, new_start: Any) -> None:
        self.start = new_start

    def update_end(self, new_end: Any) -> None:
        self.end = new_end

    def day_in_the_interval(self, entity: Entity | None = None) -> int:
        start = self.start
        if start is None:
            if entity is None:
                raise ValueError("entity is required when interval start is unknown")
            start = entity.life_span.get_start()
        end = self.end
        if self.end is None:
            if entity is None:
                raise ValueError("entity is required when interval end is unknown")
            end = entity.life_span.get_end()
        if start is None or end is None:
            raise ValueError("interval bounds cannot both remain unknown")
        delta_days = (end - start).astype("timedelta64[D]")
        return int(delta_days.astype(int))




    def is_A_before(self, other: Interval) -> bool:
        return bool(self.get_end() < other.get_start())


    def is_A_equal(self, other: Interval) -> bool:
        return bool(
            (self.get_start() == other.get_start())
            and (self.get_end() == other.get_end())
        )


    def is_A_meets(self, other: Interval) -> bool:
        return bool(self.get_end() == other.get_start())


    def is_A_overlaps(self, other: Interval) -> bool:
        return bool(
            (self.get_start() < other.get_start())
            and (self.get_end() > other.get_start())
            and (self.get_end() < other.get_end())
        )


    def is_A_during(self, other: Interval) -> bool:
        return bool(
            (self.get_start() > other.get_start())
            and (self.get_end() < other.get_end())
        )

    def is_A_starts(self, other: Interval) -> bool:
        return bool(
            (self.get_start() == other.get_start())
            and (self.get_end() < other.get_end())
        )

    def is_A_finishes(self, other: Interval) -> bool:
        return bool(
            (self.get_end() == other.get_end())
            and (self.get_start() > other.get_start())
        )


    def is_A_verification(
        self, other: Interval
    ) -> dict[
        str, bool
    ]:
        return {
            "before": self.is_A_before(other),
            "equal": self.is_A_equal(other),
            "meets": self.is_A_meets(other),
            "overlaps": self.is_A_overlaps(other),
            "during": self.is_A_during(other),
            "starts": self.is_A_starts(other),
            "finishes": self.is_A_finishes(other),
        }


class Triple:
    head: str
    relation: str
    value: Any
    date: Interval
    relation_type: str

    def __init__(self, head: str, relation: str, value: Any, date: Interval) -> None:
        self.head = head
        self.relation = relation
        self.value = value
        self.date = date

        if isinstance(value, str) and (
            (value[: len("http://")] == "http://")
            or (value[: len("https://")] == "https://")
        ):
            self.relation_type = "Object"
        else:
            self.relation_type = "Datatype"

    def __str__(self) -> str:
        return str((self.head, self.relation, self.value, self.date))

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.head, self.relation, self.value, str(self.date)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return NotImplemented
        return (
            (self.head == other.head)
            and (self.relation == other.relation)
            and (self.value == other.value)
            and (self.date == other.date)
            and (self.relation_type == other.relation_type)
        )


class Entity:
    name: str
    life_span: Interval
    triples_per_p: dict[str, set[Triple]]
    triples_per_r_and_rxv: dict[Any, set[Triple]]
    today: np.datetime64
    granularity: str

    def __init__(self, name: str, today: np.datetime64, granularity: str) -> None:
        self.name = name
        self.life_span = Interval(None, None)
        self.triples_per_p = {}
        self.triples_per_r_and_rxv = {}
        self.today = today
        self.granularity = granularity

    def __str__(self) -> str:
        return str((self.name, str(self.life_span)))

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def add_triple(self, triple: Triple) -> None:

        if triple.relation not in self.triples_per_p:
            self.triples_per_p[triple.relation] = set()
            self.triples_per_r_and_rxv[triple.relation] = set()

        self.triples_per_p[triple.relation].add(triple)
        self.triples_per_r_and_rxv[triple.relation].add(triple)


        prefix = "http://"
        if isinstance(triple.value, str) and triple.value[: len(prefix)] == prefix:
            if (triple.relation, triple.value) not in self.triples_per_r_and_rxv:
                self.triples_per_r_and_rxv[(triple.relation, triple.value)] = set()

            self.triples_per_r_and_rxv[(triple.relation, triple.value)].add(triple)

        if triple.date.get_start() is not None:
            if (self.life_span.get_start() is not None) and (
                self.life_span.get_start() > triple.date.get_start()
            ):
                self.life_span.update_start(new_start=triple.date.get_start())
            elif self.life_span.get_start() is None:
                self.life_span.update_start(new_start=triple.date.get_start())

            if (self.life_span.get_end() is not None) and (
                self.life_span.get_end() < triple.date.get_start()
            ):
                self.life_span.update_end(
                    new_end=triple.date.get_start()
                    + np.timedelta64(1, self.granularity)
                )
            elif self.life_span.get_end() is None:
                self.life_span.update_end(
                    new_end=triple.date.get_start()
                    + np.timedelta64(1, self.granularity)
                )

        if triple.date.get_end() is not None:
            if (self.life_span.get_end() is not None) and (
                self.life_span.get_end() < triple.date.get_end()
            ):
                self.life_span.update_end(new_end=triple.date.get_end())
            elif self.life_span.get_end() is None:
                self.life_span.update_end(new_end=triple.date.get_end())

            if (self.life_span.get_start() is not None) and (
                self.life_span.get_start() > triple.date.get_end()
            ):
                self.life_span.update_start(
                    new_start=triple.date.get_end()
                    - np.timedelta64(1, self.granularity)
                )
            elif self.life_span.get_start() is None:
                self.life_span.update_start(
                    new_start=triple.date.get_end()
                    - np.timedelta64(1, self.granularity)
                )
        else:
            self.life_span.update_end(self.today)

    def remove_triple(self, triple: Triple) -> None:

        if triple.relation in self.triples_per_p:
            self.triples_per_p[triple.relation].remove(triple)
            self.triples_per_r_and_rxv[triple.relation].remove(triple)

            if len(self.triples_per_p[triple.relation]) == 0:
                self.triples_per_p.pop(triple.relation)
                self.triples_per_r_and_rxv.pop(triple.relation)

        if (triple.relation, triple.value) in self.triples_per_r_and_rxv:
            self.triples_per_r_and_rxv[(triple.relation, triple.value)].remove(triple)

            if len(self.triples_per_r_and_rxv[(triple.relation, triple.value)]) == 0:
                self.triples_per_r_and_rxv.pop((triple.relation, triple.value))

    def update_lifespan(self) -> None:

        new_early = None
        new_latest = None

        for r in self.triples_per_p:
            for t in self.triples_per_p[r]:
                if new_early is None and t.date.get_start() is not None:
                    new_early = t.date.get_start()
                elif t.date.get_start() is not None and new_early > t.date.get_start():
                    new_early = t.date.get_start()
                elif t.date.get_end() is not None and (
                    new_early is None
                    or new_early
                    > t.date.get_end() - np.timedelta64(1, self.granularity)
                ):
                    new_early = t.date.get_end() - np.timedelta64(1, self.granularity)

                if t.date.get_end() is None:
                    new_latest = self.today
                elif new_latest is None:
                    new_latest = t.date.get_end()
                elif new_latest < t.date.get_end():
                    new_latest = t.date.get_end()

        self.life_span = Interval(new_early, new_latest)

    def get_lifespan(self) -> Interval:
        return self.life_span

    def get_number_of_days(self) -> int:
        return self.life_span.day_in_the_interval(self)

    def get_triples_with_p(
        self, p: str
    ) -> set[Triple] | None:
        if p in self.triples_per_p:
            return self.triples_per_p[p]
        return None

    def generate_triples_per_r_and_rxv(self, rxv_allowed: set[tuple[str, Any]]) -> None:
        self.triples_per_r_and_rxv = copy.deepcopy(self.triples_per_p)

        for r in self.triples_per_p:
            for t in self.triples_per_p[r]:
                v = t.value
                if (r, v) in rxv_allowed:
                    if (r, v) not in self.triples_per_r_and_rxv:
                        self.triples_per_r_and_rxv[(r, v)] = set()
                    self.triples_per_r_and_rxv[(r, v)].add(t)

    def generate_all_triples_per_r_and_rxv(self) -> None:
        self.triples_per_r_and_rxv = copy.deepcopy(self.triples_per_p)

        for r in self.triples_per_p:
            for t in self.triples_per_p[r]:
                v = t.value
                if (r, v) not in self.triples_per_r_and_rxv:
                    self.triples_per_r_and_rxv[(r, v)] = set()
                self.triples_per_r_and_rxv[(r, v)].add(t)


class TimeSequence:
    intervals: list[Interval]
    multi_valuation_temporal: bool
    meets: int

    def __init__(self, intervals: list[Interval]) -> None:

        self.intervals = intervals
        self.multi_valuation_temporal = False
        self.meets = 0

        for i in range(len(intervals) - 1):
            if (
                intervals[i].get_end() > intervals[i + 1].get_start()
            ):
                self.multi_valuation_temporal = (
                    True
                )

            if self.intervals[i].is_A_meets(
                self.intervals[i + 1]
            ):
                self.meets += 1

    def __len__(self) -> int:
        return len(self.intervals)


    def find_inter_comparison(
        self, other: TimeSequence
    ) -> list[
        tuple[Interval, Interval]
    ]:
        if (
            self.multi_valuation_temporal or other.multi_valuation_temporal
        ):
            print("Working on multi")

        if (len(self.intervals) == 0) or (
            len(other.intervals) == 0
        ):
            print("One is len == 0")
            return []





        def find_next_earliest_interval(
            last_index_per_seq: Any,
        ) -> tuple[int, Any, Any]:
            self_index, other_index = last_index_per_seq


            if (self_index + 1 >= len(self.intervals)) and (
                other_index + 1 >= len(other.intervals)
            ):
                return -1, None, None
            elif self_index + 1 >= len(self.intervals):
                return (
                    1,
                    [self_index, other_index + 1],
                    other.intervals[other_index + 1],
                )
            elif other_index + 1 >= len(other.intervals):
                return 0, [self_index + 1, other_index], self.intervals[self_index + 1]

            if (
                self.intervals[self_index + 1].get_start()
                < other.intervals[other_index + 1].get_start()
            ):
                return 0, [self_index + 1, other_index], self.intervals[self_index + 1]
            elif (
                self.intervals[self_index + 1].get_start()
                > other.intervals[other_index + 1].get_start()
            ):
                return (
                    1,
                    [self_index, other_index + 1],
                    other.intervals[other_index + 1],
                )
            elif (
                self.intervals[self_index + 1].get_start()
                == other.intervals[other_index + 1].get_start()
            ):
                return (
                    2,
                    [self_index + 1, other_index + 1],
                    (self.intervals[self_index + 1], other.intervals[other_index + 1]),
                )
            return -1, None, None



        def which_cursor_to_move(c_0: Any, c_1: Any) -> int:
            if c_0[2].get_end() < c_1[2].get_end():
                return 0
            elif c_0[2].get_end() > c_1[2].get_end():
                return 1
            return 2

        last_index_per_seq: Any = [-1, -1]

        c_0: Any = (None, None, None)
        c_1: Any = (None, None, None)


        name_sequence, last_index_per_seq, interval = find_next_earliest_interval(
            last_index_per_seq
        )
        if name_sequence == -1:
            return []
        elif name_sequence != 2:
            c_0 = (last_index_per_seq[name_sequence], name_sequence, interval)
        else:
            c_0 = (last_index_per_seq[0], 0, interval[0])
            c_1 = (last_index_per_seq[1], 1, interval[1])


        if c_1[0] is None:
            name_sequence, last_index_per_seq, interval = find_next_earliest_interval(
                last_index_per_seq
            )
            if name_sequence != 2:
                c_1 = (last_index_per_seq[name_sequence], name_sequence, interval)
            else:
                c_0 = (last_index_per_seq[0], 0, interval[0])
                c_1 = (last_index_per_seq[1], 1, interval[1])

        comparisons: list[tuple[Interval, Interval]] = []
        while True:
            if c_0[1] != c_1[1]:
                if c_0[1] == 0:
                    comparisons.append((c_0[2], c_1[2]))
                else:
                    comparisons.append((c_1[2], c_0[2]))

            c_to_move = which_cursor_to_move(c_0, c_1)
            if c_to_move == 0:
                name_sequence, last_index_per_seq, interval = (
                    find_next_earliest_interval(last_index_per_seq)
                )
                if name_sequence == -1:
                    break
                elif name_sequence != 2:
                    c_0 = (last_index_per_seq[name_sequence], name_sequence, interval)
                else:
                    c_0 = (last_index_per_seq[0], 0, interval[0])
                    c_1 = (last_index_per_seq[1], 1, interval[1])

            elif c_to_move == 1:
                name_sequence, last_index_per_seq, interval = (
                    find_next_earliest_interval(last_index_per_seq)
                )
                if name_sequence == -1:
                    break
                elif name_sequence != 2:
                    c_1 = (last_index_per_seq[name_sequence], name_sequence, interval)
                else:
                    c_0 = (last_index_per_seq[0], 0, interval[0])
                    c_1 = (last_index_per_seq[1], 1, interval[1])


            else:
                name_sequence, last_index_per_seq, interval = (
                    find_next_earliest_interval(last_index_per_seq)
                )
                if name_sequence == -1:
                    break
                elif name_sequence != 2:
                    c_0 = (last_index_per_seq[name_sequence], name_sequence, interval)

                    name_sequence, last_index_per_seq, interval = (
                        find_next_earliest_interval(last_index_per_seq)
                    )
                    if name_sequence == -1:
                        break
                    elif name_sequence != 2:
                        c_1 = (
                            last_index_per_seq[name_sequence],
                            name_sequence,
                            interval,
                        )
                    else:
                        c_0 = (last_index_per_seq[0], 0, interval[0])
                        c_1 = (last_index_per_seq[1], 1, interval[1])
                else:
                    c_0 = (last_index_per_seq[0], 0, interval[0])
                    c_1 = (last_index_per_seq[1], 1, interval[1])

        return comparisons


class TimeSequenceRelation:
    name_axioms: set[str] = {
        "equal",
        "before",
        "meets",
        "overlaps",
        "during",
        "starts",
        "finishes",
    }

    name_relation_A: str
    name_relation_B: str

    seq_A: TimeSequence
    seq_B: TimeSequence


    inter_comparison_A_to_B: dict[
        str, int
    ]
    inter_comparison_B_to_A: dict[
        str, int
    ]

    A_o_B: set[str]
    B_o_A: set[str]

    verified: bool

    def __init__(
        self,
        name_relation_A: str,
        name_relation_B: str,
        seq_A: TimeSequence,
        seq_B: TimeSequence,
        constraint_to_check: TemporalRule | None = None,
    ) -> None:
        name_relation_A = str(
            name_relation_A
        )
        name_relation_B = str(
            name_relation_B
        )

        if (
            name_relation_A < name_relation_B
        ):
            self.name_relation_A = name_relation_A
            self.name_relation_B = name_relation_B

            self.seq_A = seq_A
            self.seq_B = seq_B

        else:
            self.name_relation_A = name_relation_B
            self.name_relation_B = name_relation_A

            self.seq_A = seq_B
            self.seq_B = seq_A

        compare = self.seq_A.find_inter_comparison(
            self.seq_B
        )

        inter_raw_a_to_b = [
            int_a.is_A_verification(int_b) for (int_a, int_b) in compare
        ]
        inter_raw_b_to_a = [
            int_b.is_A_verification(int_a) for (int_a, int_b) in compare
        ]

        self.inter_comparison_A_to_B = {
            key: sum([inter_raw_a_to_b[i][key] for i in range(len(inter_raw_a_to_b))])
            for key in self.name_axioms
        }


        self.inter_comparison_B_to_A = {
            key: sum([inter_raw_b_to_a[i][key] for i in range(len(inter_raw_b_to_a))])
            for key in self.name_axioms
        }


        self.A_o_B = set()
        self.B_o_A = set()

        if not constraint_to_check:
            self.verify_axioms_props()
            self.verify_multi_axioms_props()

        else:
            self.verified = self.apply_only_constraint(constraint_to_check)

    def get_name(self) -> tuple[str, str]:
        return (self.name_relation_A, self.name_relation_B)



    def is_A_equal_B(self):
        if self.inter_comparison_A_to_B["equal"] == len(self.seq_A.intervals):
            return "equal_axiom"
        return ""

    def is_A_before_B(self):
        if self.inter_comparison_A_to_B["before"] == len(self.seq_A.intervals):
            return "before"
        return ""

    def is_A_meets_B(self):
        if self.inter_comparison_A_to_B["meets"] == len(self.seq_A.intervals):
            return "meets"
        return ""

    def is_A_overlaps_B(self):
        if self.inter_comparison_A_to_B["overlaps"] == len(self.seq_A.intervals):
            return "overlaps"
        return ""

    def is_A_during_B(self):
        if self.inter_comparison_A_to_B["during"] == len(self.seq_A.intervals):
            return "during"
        return ""

    def is_A_starts_B(self):
        if self.inter_comparison_A_to_B["starts"] == len(self.seq_A.intervals):
            return "starts"
        return ""

    def is_A_finishes_B(self):
        if self.inter_comparison_A_to_B["finishes"] == len(self.seq_A.intervals):
            return "finishes"
        return ""

    def is_B_equal_A(self):
        if self.inter_comparison_B_to_A["equal"] == len(self.seq_B.intervals):
            return "equal_axiom"
        return ""

    def is_B_before_A(self):
        if self.inter_comparison_B_to_A["before"] == len(self.seq_B.intervals):
            return "before"
        return ""

    def is_B_meets_A(self):
        if self.inter_comparison_B_to_A["meets"] == len(self.seq_B.intervals):
            return "meets"
        return ""

    def is_B_overlaps_A(self):
        if self.inter_comparison_B_to_A["overlaps"] == len(self.seq_B.intervals):
            return "overlaps"
        return ""

    def is_B_during_A(self):
        if self.inter_comparison_B_to_A["during"] == len(self.seq_B.intervals):
            return "during"
        return ""

    def is_B_starts_A(self):
        if self.inter_comparison_B_to_A["starts"] == len(self.seq_B.intervals):
            return "starts"
        return ""

    def is_B_finishes_A(self):
        if self.inter_comparison_B_to_A["finishes"] == len(self.seq_B.intervals):
            return "finishes"
        return ""

    def are_equal(self):
        return (
            self.inter_comparison_A_to_B["equal"] == len(self.seq_A.intervals)
        ) and (self.inter_comparison_B_to_A["equal"] == len(self.seq_B.intervals))

    def verify_axioms_props(self):

        if self.are_equal():
            self.A_o_B.add("are equals")
            self.B_o_A.add("are equals")

        fct_A_o_B = [
            self.is_A_equal_B,
            self.is_A_before_B,
            self.is_A_meets_B,
            self.is_A_overlaps_B,
            self.is_A_during_B,
            self.is_A_starts_B,
            self.is_A_finishes_B,
        ]

        for fct in fct_A_o_B:
            res = fct()
            if res:
                self.A_o_B.add(res)

        fct_B_o_A = [
            self.is_B_equal_A,
            self.is_B_before_A,
            self.is_B_meets_A,
            self.is_B_overlaps_A,
            self.is_B_during_A,
            self.is_B_starts_A,
            self.is_B_finishes_A,
        ]

        for fct in fct_B_o_A:
            res = fct()
            if res:
                self.B_o_A.add(res)




    def check_axiom_validity(self, axiom_name: str) -> bool:
        if axiom_name == "equal":
            return (
                self.inter_comparison_A_to_B["equal"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["equal"] == len(self.seq_B.intervals))
        elif axiom_name == "before":
            return (
                self.inter_comparison_A_to_B["before"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["before"] == len(self.seq_B.intervals))
        elif axiom_name == "meets":
            print("we are here in meets")
            return (
                self.inter_comparison_A_to_B["meets"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["meets"] == len(self.seq_B.intervals))
        elif axiom_name == "overlaps":
            return (
                self.inter_comparison_A_to_B["overlaps"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["overlaps"] == len(self.seq_B.intervals))
        elif axiom_name == "during":
            return (
                self.inter_comparison_A_to_B["during"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["during"] == len(self.seq_B.intervals))
        elif axiom_name == "starts":
            return (
                self.inter_comparison_A_to_B["starts"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["starts"] == len(self.seq_B.intervals))
        elif axiom_name == "finishes":
            return (
                self.inter_comparison_A_to_B["finishes"] == len(self.seq_A.intervals)
            ) or (self.inter_comparison_B_to_A["finishes"] == len(self.seq_B.intervals))
        elif axiom_name == "are equals":
            return (
                self.inter_comparison_A_to_B["equal"] == len(self.seq_A.intervals)
            ) and (self.inter_comparison_B_to_A["equal"] == len(self.seq_B.intervals))
        else:
            return False

    def is_either_one_or_the_other(self):


        sum_axioms = 0
        for axiom in self.name_axioms.difference({"before", "meets"}):
            sum_axioms += self.inter_comparison_A_to_B[axiom]
            sum_axioms += self.inter_comparison_B_to_A[axiom]

        if sum_axioms == 0:
            return "either_one_or_the_other"

        return ""

    def is_one_or_the_other_not_closed(self):


        if self.seq_A.meets + self.seq_B.meets != 0:
            return ""

        sum_axioms = 0
        for axiom in self.name_axioms.difference({"before"}):
            sum_axioms += self.inter_comparison_A_to_B[axiom]
            sum_axioms += self.inter_comparison_B_to_A[axiom]

        if sum_axioms == 0:
            return "one_or_the_other_not_closed"
        return ""

    def is_one_or_the_other_closed(self):


        sum_axioms = 0
        for axiom in self.name_axioms.difference({"meets"}):
            sum_axioms += self.inter_comparison_A_to_B[axiom]
            sum_axioms += self.inter_comparison_B_to_A[axiom]

        if sum_axioms == 0:
            return "one_or_the_other_closed"
        return ""

    def is_in_between(self):
        if (
            self.inter_comparison_A_to_B["before"]
            + self.inter_comparison_B_to_A["before"]
        ) == (len(self.seq_A.intervals) + len(self.seq_B.intervals) - 1):
            return "in_between"
        return ""

    def is_in_between_closed(self):
        if (
            self.inter_comparison_A_to_B["meets"]
            + self.inter_comparison_B_to_A["meets"]
        ) == (len(self.seq_A.intervals) + len(self.seq_B.intervals) - 1):
            return "in_between_closed"
        return ""

    def is_seq_A_before_seq_B(self):
        if (
            (self.inter_comparison_A_to_B["before"] == 1)
            & (sum(self.inter_comparison_A_to_B.values()) == 1)
            & (sum(self.inter_comparison_B_to_A.values()) == 0)
        ):
            return "sequence_before"
        return ""

    def is_seq_B_before_seq_A(self):
        if (
            (self.inter_comparison_B_to_A["before"] == 1)
            & (sum(self.inter_comparison_B_to_A.values()) == 1)
            & (sum(self.inter_comparison_A_to_B.values()) == 0)
        ):
            return "sequence_before"
        return ""

    def is_seq_A_meets_seq_B(self):
        if (
            (self.inter_comparison_A_to_B["meets"] == 1)
            & (sum(self.inter_comparison_A_to_B.values()) == 1)
            & (sum(self.inter_comparison_B_to_A.values()) == 0)
        ):
            return "sequence_meets"
        return ""

    def is_seq_B_meets_seq_A(self):
        if (
            (self.inter_comparison_B_to_A["meets"] == 1)
            & (sum(self.inter_comparison_B_to_A.values()) == 1)
            & (sum(self.inter_comparison_A_to_B.values()) == 0)
        ):
            return "sequence_meets"
        return ""

    def is_seq_A_always_with_seq_B(self):
        if (
            self.inter_comparison_A_to_B["equal"]
            + self.inter_comparison_A_to_B["during"]
            + self.inter_comparison_A_to_B["starts"]
            + self.inter_comparison_A_to_B["finishes"]
        ) == len(self.seq_A):
            return "always_with"
        return ""

    def is_seq_B_always_with_seq_A(self):
        if (
            self.inter_comparison_B_to_A["equal"]
            + self.inter_comparison_B_to_A["during"]
            + self.inter_comparison_B_to_A["starts"]
            + self.inter_comparison_B_to_A["finishes"]
        ) == len(self.seq_B):
            return "always_with"
        return ""

    def is_always_overlapping(self):
        if (
            self.inter_comparison_A_to_B["overlaps"]
            + self.inter_comparison_B_to_A["overlaps"]
        ) == len(self.seq_A) + len(self.seq_B) - 1:
            return "always_overlapping"
        return ""











    def verify_multi_axioms_props(self) -> None:

        fcts = [
            self.is_either_one_or_the_other,
            self.is_one_or_the_other_not_closed,
            self.is_one_or_the_other_closed,
            self.is_in_between,
            self.is_in_between_closed,
            self.is_always_overlapping,
        ]

        fcts_A_o_B = [
            self.is_seq_A_before_seq_B,
            self.is_seq_A_meets_seq_B,
            self.is_seq_A_always_with_seq_B,
        ]

        fcts_B_o_A = [
            self.is_seq_B_before_seq_A,
            self.is_seq_B_meets_seq_A,
            self.is_seq_B_always_with_seq_A,
        ]
        for fct in fcts:
            res = fct()
            if res:
                self.A_o_B.add(res)
                self.B_o_A.add(res)

        for fct in fcts_A_o_B:
            res = fct()
            if res:
                self.A_o_B.add(res)

        for fct in fcts_B_o_A:
            res = fct()
            if res:
                self.B_o_A.add(res)

    def verify_multi_axioms_props_restrictif(self) -> None:







        res = self.is_seq_A_before_seq_B()
        if res:
            self.A_o_B.add(res)
            return

        res = self.is_seq_B_before_seq_A()
        if res:
            self.B_o_A.add(res)
            return



        res = self.is_in_between()
        if res:
            self.A_o_B.add(res)
            self.B_o_A.add(res)
            return



        res = self.is_in_between_closed()
        if res:
            self.A_o_B.add(res)
            self.B_o_A.add(res)
            return



        res = self.is_one_or_the_other_not_closed()
        if res:
            self.A_o_B.add(res)
            self.B_o_A.add(res)
            return



        res = self.is_one_or_the_other_closed()
        if res:
            self.A_o_B.add(res)
            self.B_o_A.add(res)
            return



        res = self.is_either_one_or_the_other()
        if res:
            self.A_o_B.add(res)
            self.B_o_A.add(res)
            return

    def apply_only_constraint(self, constraint: TemporalRule) -> bool:
        nc = constraint.get_r()

        if nc == "either_one_or_the_other":
            return self.is_either_one_or_the_other() != ""
        elif nc == "one_or_the_other_not_closed":
            return self.is_one_or_the_other_not_closed() != ""
        elif nc == "one_or_the_other_closed":
            return self.is_one_or_the_other_closed() != ""
        elif nc == "in_between_closed":
            return self.is_in_between_closed() != ""
        elif nc == "in_between":
            return self.is_in_between() != ""
        elif nc == "always_overlapping":
            return self.is_always_overlapping() != ""

        if constraint.get_a() == self.name_relation_A:
            if nc == "equal_axiom":
                return self.is_A_equal_B() != ""
            elif nc == "before":
                return self.is_A_before_B() != ""
            elif nc == "meets":
                return self.is_A_meets_B() != ""
            elif nc == "overlaps":
                return self.is_A_overlaps_B() != ""
            elif nc == "during":
                return self.is_A_during_B() != ""
            elif nc == "starts":
                return self.is_A_starts_B() != ""
            elif nc == "finishes":
                return self.is_A_finishes_B() != ""
            elif nc == "sequence_before":
                return self.is_seq_A_before_seq_B() != ""
            elif nc == "sequence_meets":
                return self.is_seq_A_meets_seq_B() != ""
            elif nc == "always_with":
                return self.is_seq_A_always_with_seq_B() != ""
            else:
                print(f"I am not handled : {constraint}")
                return False

        else:
            if nc == "equal_axiom":
                return self.is_B_equal_A() != ""
            elif nc == "before":
                return self.is_B_before_A() != ""
            elif nc == "meets":
                return self.is_B_meets_A() != ""
            elif nc == "overlaps":
                return self.is_B_overlaps_A() != ""
            elif nc == "during":
                return self.is_B_during_A() != ""
            elif nc == "starts":
                return self.is_B_starts_A() != ""
            elif nc == "finishes":
                return self.is_B_finishes_A() != ""
            elif nc == "sequence_before":
                return self.is_seq_B_before_seq_A() != ""
            elif nc == "sequence_meets":
                return self.is_seq_B_meets_seq_A() != ""
            elif nc == "always_with":
                return self.is_seq_B_always_with_seq_A() != ""
            else:
                print(f"I am not handled : {constraint}")
                return False


class TemporalRule:
    a: str
    precision_a: str | None
    r: str
    b: str
    precision_b: str | None

    error_percentage: float
    coverage_percentage: float

    def __init__(
        self,
        a: str,
        precision_a: str | None,
        r: str,
        b: str,
        precision_b: str | None,
        error_percentage: float,
        coverage_percentage: float,
    ) -> None:

        self.a = a
        self.precision_a = precision_a

        self.r = r

        self.b = b
        self.precision_b = precision_b

        self.error_percentage = error_percentage
        self.coverage_percentage = coverage_percentage

    def __str__(self) -> str:
        res = ""
        if not self.precision_a:
            res += f"{self.a} "
        else:
            res += f"{self.a} X {self.precision_a} "

        res += f"{self.r} "

        if not self.precision_b:
            res += f"{self.b} "
        else:
            res += f"{self.b} X {self.precision_b} "

        return res + f": [e:{self.error_percentage}, c:{self.coverage_percentage}]"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        res = self.a
        if self.precision_a:
            res += "X" + self.precision_a
        res += " " + self.r + " "
        res += self.b
        if self.precision_b:
            res += "X" + self.precision_b
        return hash(res)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporalRule):
            return NotImplemented
        return (
            self.a == other.a
            and self.precision_a == other.precision_a
            and self.r == other.r
            and self.b == other.b
            and self.precision_b == other.precision_b
        )







    def get_a(self) -> str | tuple[str, str]:
        if not self.precision_a:
            return self.a
        else:
            return (self.a, self.precision_a)







    def get_b(self) -> str | tuple[str, str]:
        if not self.precision_b:
            return self.b
        else:
            return (self.b, self.precision_b)

    def get_r(self) -> str:
        return self.r

    def to_tsv(self) -> str:
        res = ""

        if not self.precision_a:
            res += f"{self.a}\t"
        else:
            res += f"({self.a}, {self.precision_a})\t"

        res += f"{self.r}\t"

        if not self.precision_b:
            res += f"{self.b}\t"
        else:
            res += f"({self.b}, {self.precision_b})\t"

        return res + f"{self.error_percentage}\t{self.coverage_percentage}"

    @staticmethod
    def load_a_rule(line: str) -> TemporalRule:
        line_splited = line.strip().split("\t")
        if len(line_splited) != 5:
            raise ValueError("ERROR LOAD NOT A TEMPORAL RULE LINE")

        def parse_rule_part(raw_part: str) -> tuple[str, str | None]:
            part = raw_part.strip()
            if part.startswith("(") and part.endswith(")") and ", " in part:
                relation, precision = part[1:-1].split(", ", maxsplit=1)
                return relation, precision
            return part, None

        a, precision_a = parse_rule_part(line_splited[0])
        b, precision_b = parse_rule_part(line_splited[2])
        return TemporalRule(
            a,
            precision_a,
            line_splited[1],
            b,
            precision_b,
            float(line_splited[3]),
            float(line_splited[4]),
        )

    def is_useful_for_e(self, e: Entity) -> bool:

        if not self.precision_a:
            if not e.get_triples_with_p(self.a):
                return False
        else:
            triples = e.get_triples_with_p(self.a)
            if not triples:
                return False
            if not any(t.value == self.precision_a for t in triples):
                return False


        if not self.precision_b:
            if not e.get_triples_with_p(self.b):
                return False
        else:
            triples = e.get_triples_with_p(self.b)
            if not triples:
                return False
            if not any(t.value == self.precision_b for t in triples):
                return False


        return True




def ordered_time_sequence_first_start(entity: Entity, r: str) -> list[Interval]:
    triples_of_r = entity.get_triples_with_p(r)
    if triples_of_r is not None:
        r_per_start = {}
        for t in triples_of_r:
            start_triple, end_triple = t.date.get_start(), t.date.get_end()

            if start_triple is None:
                start_triple = entity.get_lifespan().get_start()

            if end_triple is None:
                end_triple = entity.get_lifespan().get_end()

            if start_triple not in r_per_start:
                r_per_start[start_triple] = set()

            r_per_start[start_triple].add(Interval(start_triple, end_triple))

        return [i for start in sorted(r_per_start.keys()) for i in r_per_start[start]]

    return []


def ordered_time_sequence_first_start_with_rxv(
    entity: Entity, r: str | tuple[str, Any]
) -> list[Interval]:
    if r in entity.triples_per_r_and_rxv:
        r_per_start = {}
        triples_of_r = entity.triples_per_r_and_rxv[r]
        for t in triples_of_r:
            start_triple, end_triple = t.date.get_start(), t.date.get_end()

            if start_triple is None:
                start_triple = entity.get_lifespan().get_start()

            if end_triple is None:
                end_triple = entity.get_lifespan().get_end()

            if start_triple not in r_per_start:
                r_per_start[start_triple] = set()

            r_per_start[start_triple].add(Interval(start_triple, end_triple))

        return [i for start in sorted(r_per_start.keys()) for i in r_per_start[start]]

    return []
