#pragma once

struct list_node
{
    struct list_node *next, *prev;
};

#define LIST_INIT(name) { &(name), &(name) }

#define LIST(name) struct list_node name = LIST_INIT(name)

static inline void __list_add(struct list_node *news,
                    struct list_node *prev,
                    struct list_node *next)
{
    next->prev = news;
    news->next = next;
    news->prev = prev;
    prev->next = news;
}

static inline void push_front(struct list_node *news, struct list_node *head)
{
    __list_add(news, head, head->next);
}

static inline void my_push_back(struct list_node *news, struct list_node *head)
{
    __list_add(news, head->prev, head);
}

#define offsetof(TYPE, MEMBER) ((size_t)&((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({ void *__mptr = (void *)(ptr); ((type *)(__mptr - offsetof(type, member))); })

#define list_entry(ptr, type, member) container_of(ptr, type, member)

#define list_first_entry(ptr, type, member) list_entry((ptr)->next, type, member)

#define list_last_entry(ptr, type, member) list_entry((ptr)->prev, type, member)

#define list_for_each(pos, head) for (pos = (head)->next; pos != (head); pos = pos->next)