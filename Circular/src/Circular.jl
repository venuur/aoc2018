module Circular

export circular_list, insertafter!, insertbefore!, next, prev, value,
    removeleftshift!

mutable struct DoublyLinkedList{T}
    value::T
    prev::Union{DoublyLinkedList{T}, Nothing}
    next::Union{DoublyLinkedList{T}, Nothing}
end

DoublyLinkedList(elem::T) where T = DoublyLinkedList(elem, nothing, nothing)

mutable struct CircularList{T}
    head::DoublyLinkedList{T}
    function CircularList{T}(ddl::DoublyLinkedList) where T
        cl = new(ddl)
    end
end

wrap_ddl(ddl::DoublyLinkedList{T}) where T = CircularList{T}(ddl)

function circular_list(a::T) where T
    cl = CircularList{T}(DoublyLinkedList(a))
    cl.head.next = cl.head
    cl.head.prev = cl.head
    cl
end

function circular_list(ddl::DoublyLinkedList{T}) where T
    cl = CircularList{T}(dd)
    back = cl.head
    while back.next !== nothing && back.next !== cl.head
        back = back.next
    end
    front = cl.head
    while front.prev !== nothing && front.prev !== back
        front = front.prev
    end
    front.prev = back
    back.next = front
    cl
end


function Base.insert!(cl::CircularList{T}, i, v::T) where T
    if i == 0
        insertbefore!(cl, v)
    elseif i > 0
        this = cl.head
        for j in 1:i
            this = this.next
        end
        insertbefore!(this, v)
    else
        this = cl.head
        for j = 1:-i
            this = this.prev
        end
        insertbefore!(this, v)
    end
end

function insertafter!(cl::CircularList{T}, v::T) where T
    ins = DoublyLinkedList(v)
    after = cl.head.next
    after.prev = ins
    cl.head.next = ins
    ins.prev = cl.head
    ins.next = after
    wrap_ddl(ins)
end

function insertbefore!(cl::CircularList{T}, v::T) where T
    before = cl.head.prev
    ins = DoublyLinkedList{}(v, before, cl.head)
    cl.head.prev = ins
    before.next = ins
    wrap_ddl(ins)
end

function removeleftshift!(cl::CircularList{T}) where T
    here = cl.head
    before = here.prev
    after = here.next
    before.next = after
    after.prev = before
    here = nothing
    cl.head = after
    cl
end

function next(cl::CircularList{T}) where T
    wrap_ddl(cl.head.next)
end

function prev(cl::CircularList{T}) where T
    wrap_ddl(cl.head.prev)
end

function value(cl::CircularList{T}) where T
    cl.head.value
end

function Base.show(io::IO, cl::CircularList{T}) where T
    head = cl.head
    print(io, "[")
    show(io, head.value)
    next = head.next
    while next !== head
        print(io, ",")
        show(io, next.value)
        next = next.next
    end
    print(io, "]")
end

end
